/*******************************************************************************
* Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#pragma once

#include <iostream>


namespace zendnn {

//class to read environment variables for zendnnn
//In future this will be used with operator memory desc
class zendnnEnv {
  public:
    uint    omp_num_threads;
    uint    zen_num_threads;
    uint    zenGEMMalgo;
    bool    zenBlockedFormat;
    bool    zenBlockedNHWC;
    uint    zenEnableMemPool;
    bool    zenLibMemPoolEnable;
    bool    zenINT8format;

    //setting default values
    zendnnEnv() {
        omp_num_threads = 1;
        zen_num_threads = 1;
        zenGEMMalgo = 0;
        zenBlockedFormat = false;
        zenBlockedNHWC = false;
        zenEnableMemPool = 1;
        zenLibMemPoolEnable = true;
        zenINT8format = false;
    }
};

/// Read an integer from the environment variable
/// Return default_value if the environment variable is not defined, otherwise
/// return actual value.
inline int zendnn_getenv_int(const char *name, int default_value = 0) {
    char *val = std::getenv(name);
    return val == NULL ? default_value : atoi(val);
}

/// Read an float from the environment variable
/// Return default_value if the environment variable is not defined, otherwise
/// return actual value.
inline float zendnn_getenv_float(const char *name, float default_value = 0.0f) {
    char *val = std::getenv(name);
    return val == NULL ? default_value : atof(val);
}

/// Read an string from the environment variable
/// Return empty string "" if the environment variable is not defined, otherwise
/// return actual value.
inline std::string zendnn_getenv_string(const char *name,
                                        std::string default_value = "") {
    char *val = std::getenv(name);
    return val == NULL ? default_value : std::string(val);
}

}

zendnn::zendnnEnv readEnv();

extern "C" {
    void zenConvolution2D(
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
        const int out_width,
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0
    );

    void zenConvolution2DwithBias(
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

    void zenConvolution2DwithBiasRelu(
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

    void zenConvolution2DwithBatchNorm(
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
        const int out_width,
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0
    );

    void zenConvolution2DwithBatchNormRelu(
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
        const int out_width,
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0
    );

    void zenConvolution2DwithBatchNormsum(
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
        const float *elemetwise_input,
        float *out_layer,
        const int out_height,
        const int out_width,
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0
    );

    void zenBatchMatMul(
        bool Layout,
        bool TransA,
        bool TransB,
        int *M_Array,
        int *N_Array,
        int *K_Array,
        const float *alpha_Array,
        const float **A_Array,
        int *lda_Array,
        const float **B_Array,
        int *ldb_Array,
        const float *beta_Array,
        float **C_Array,
        int *ldc_Array,
        int group_count,
        int *group_size
    );

    void max_pooling(
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

    void avg_pooling(
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

    void zenPostOps(
        zendnn::zendnnEnv zenEnvObj,
        float *out_layer,
        const float *elemtwise_input,
        const int out_height,
        const int out_width,
        const int no_of_filter,
        const int total_filters,
        unsigned long biasOffset,
        const float *bias,
        const bool relu,
        const float *scale,
        const int no_of_threads,
        const float *offset = NULL,
        const float  *mean = NULL,
        const int batch_size = 1
    );

    void zenClipOp(
        zendnn::zendnnEnv zenEnvObj,
        float *out_layer,
        float upperbound,
        unsigned long size
    );
}
