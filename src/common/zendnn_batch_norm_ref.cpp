/*******************************************************************************
* Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include "common/zendnn_private.hpp"
#include "zendnn_logging.hpp"

using namespace zendnn;
// This version implements the intial version of batch normalization
// The output from the preceding convolution is normalized , scaled and shifted

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
    const bool relu) {
    zendnnEnv zenEnvObj = readEnv();
    zendnnInfo(ZENDNN_ALGOLOG, "zenBatchNorm [zendnn batchnorm]");
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    if (data_format == 0) {       // NCHW Format
        zendnnInfo(ZENDNN_ALGOLOG, "zenBatchNorm data_format: NCHW [zendnn batchnorm]");
        if (no_of_images > 1) {
            for (int i=0; i<no_of_images; i++) {
                for (int r=0; r< no_of_filter; r++) {
                    for (int m=0; m<out_height*out_width; m++) {
                        int index = i*no_of_filter*out_height*out_width + r*out_height*out_width + m;
                        float val = scale[r]*(out_layer[index] - mean[r]) + offset[r];
                        if (relu && val < 0) {
                            out_layer[index] = 0;
                        }
                        else {
                            out_layer[index] = val;
                        }
                    }
                }
            }
        }
        else {
            for (int r=0; r< no_of_filter; r++) {
                for (int m=0; m<out_height*out_width; m++) {
                    int index = r*out_height*out_width + m;
                    float val = scale[r]*(out_layer[index] - mean[r]) + offset[r];
                    if (relu && val < 0) {
                        out_layer[index] = 0;
                    }
                    else {
                        out_layer[index] = val;
                    }
                }
            }

        }
    }
    else  {                      // NHWC Format
        zendnnInfo(ZENDNN_ALGOLOG, "zenBatchNorm data_format: NHWC [zendnn batchnorm]");
        if (no_of_images > 1) {
            for (int i=0; i<no_of_images; i++) {
                for (int m=0; m<out_height*out_width; m++) {
                    for (int r=0; r< no_of_filter; r++) {
                        int index = i*no_of_filter*out_height*out_width + m*no_of_filter + r;
                        float val = scale[r]*(out_layer[index] - mean[r]) + offset[r];
                        if (relu && val < 0) {
                            out_layer[index] = 0;
                        }
                        else {
                            out_layer[index] = val;
                        }
                    }
                }
            }
        }
        else {
            for (int m=0; m<out_height*out_width; m++) {
                for (int r=0; r< no_of_filter; r++) {
                    int index = m*no_of_filter  + r;
                    float val = scale[r]*(out_layer[index] - mean[r]) + offset[r];
                    if (relu && val < 0) {
                        out_layer[index] = 0;
                    }
                    else {
                        out_layer[index] = val;
                    }
                }
            }

        }

    }
}
