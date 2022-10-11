/*******************************************************************************
* Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#pragma once

#include <iostream>


namespace zendnn {
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
    uint    zenLibMemPoolEnable;
    bool    zenINT8format;

  private:
    //initializing ZenDNNEnv values.
    zendnnEnv() {
        omp_num_threads = zendnn_getenv_int("OMP_NUM_THREADS", 1);
        if (getenv("ZEN_NUM_THREADS")) {
            zen_num_threads = atoi(getenv("ZEN_NUM_THREADS"));
            //Overriding OMP_NUM_THREADS if ZEN_NUM_THREADS is exported
            omp_num_threads = zen_num_threads;
        }
        else {
            zen_num_threads = 1;
        }
        //ZENDNN_GEMM_ALGO is to enable specific GEMM ALGO.
        //Currently ZenDNN support three ALGO path for GEMM execution
        // If value is set to 0, library decide the optimal path
        // based on the matrix sizes and other parameter settings. However,
        // this can be overridden with specific path.
        // 1. DIRECT BLIS: MatMul is redirected to BLIS GEMM directly (zenGEMMalgo=1)
        // 2. ZenDNN+BLIS (zenGEMMalgo=2)
        //      Case 1:
        //              ZenDNN take care of problem division and thread parallelism
        //              BLIS is used for single thread GEMM execution
        //      Case 2:
        //              MatMul is redirected to BLIS directly
        // 3. ZenDNN_sgemm: zendnn_sgemm jit based kernel (zenGEMMalgo=3) (current default)
        zenGEMMalgo = zendnn_getenv_int("ZENDNN_GEMM_ALGO", 3);
        if (zenGEMMalgo<=0 || zenGEMMalgo>3) {
            zenGEMMalgo = 3;
        }
        //ZENDNN_BLOCKED_FORMAT is to enable/disable BLOCKED Format.
        zenBlockedFormat = zendnn_getenv_int("ZENDNN_BLOCKED_FORMAT", 0);
        //ZENDNN_BLOCKED_NHWC is added to support NHWC data format for CONV DIRECT ALGO
        zenBlockedNHWC = zendnn_getenv_int("ZENDNN_NHWC_BLOCKED",0);
        //ZENDNN Library gives preference to NHWC-BLOCKED Format over BLOCKED Format.
        if (zenBlockedNHWC) {
            zenBlockedFormat=0;
        }
        //TODO: change ZENDNN_ENABLE_MEMPOOL to ZENDNN_ENABLE_TF_MEMPOOL
        //use ZENDNN_ENABLE_ONNX_MEMPOOL for ONNX
        //Possible values for ZENDNN_ENABLE_MEMPOOL
        // 0 (GAM-TPA disable)
        // 1 (Graph level Memory Reuse)
        // 2 (Node level Memory Reuse)
        zenEnableMemPool = zendnn_getenv_int("ZENDNN_ENABLE_MEMPOOL", 1);
        if (zenEnableMemPool > 2) {
            zenEnableMemPool = 1;
        }
        //TODO: Unified FWK and LIB mempool for next release
        zenLibMemPoolEnable = zendnn_getenv_int("ZENDNN_ENABLE_MEMPOOL", 1);
        //ZENDNN_INT8_SUPPORT is to enable/disable INT8 support
        zenINT8format = zendnn_getenv_int("ZENDNN_INT8_SUPPORT", 0);
    }

  public:
    static const zendnnEnv &ZenDNNEnv() {
        static const zendnnEnv envObj;
        return envObj;
    }
};
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
        const int gelu,
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
