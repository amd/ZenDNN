/*******************************************************************************
* Copyright (c) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <iostream>
#include <zendnn.h>
#include <cassert>
#ifdef _WIN32
    #include <Windows.h>
    #include <chrono>
    #include <sysinfoapi.h>
    #include <corecrt_math_defines.h>
    typedef unsigned int uint;
    using namespace std::chrono;
#else
    #include <sys/sysinfo.h>
    #include <sys/time.h>
#endif

inline void *zendnn_aligned_alloc(size_t _Alignment, size_t _Size) {
#ifdef _WIN32
    return _aligned_malloc(_Size, _Alignment);
#else
    return aligned_alloc(_Alignment, _Size);
#endif
}

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
    return val == NULL ? default_value : (float)atof(val);
}

/// Read an string from the environment variable
/// Return empty string "" if the environment variable is not defined, otherwise
/// return actual value.
inline std::string zendnn_getenv_string(const char *name,
                                        std::string default_value = "") {
    char *val = std::getenv(name);
    return val == NULL ? default_value : std::string(val);
}

enum zenMatMulAlgoType {
    MATMUL_AUTO = 0,
    MATMUL_BLIS_GEMM1 = 1,
    MATMUL_BLIS_GEMM2 = 2,
    MATMUL_ZENDNN_GEMM1 = 3,
    MATMUL_ZENDNN_GEMM2 = 4,
    MATMUL_BLIS_BLOCKED_GEMM1 = 5,
    MATMUL_BLIS_BLOCKED_GEMM2 = 6,
};

enum zenBF16MatMulAlgoType {
    MATMUL_AUTO_BF16 = 0,
    MATMUL_AOCL_GEMM = 1,
    MATMUL_BLOCKED_JIT = 2,
    MATMUL_JIT = 3,
    MATMUL_AOCL_GEMM_PAR = 4,
    MATMUL_BLOCKED_JIT_PAR = 5,
    MATMUL_JIT_PAR = 6,
};

enum zenINT8MatMulAlgoType {
    MATMUL_AUTO_INT8 = 0,
    MATMUL_AOCL_GEMM_INT8 = 1,
    MATMUL_BLOCKED_JIT_INT8 = 2,
    MATMUL_JIT_INT8 = 3,
};
// enum containing all supported convolution algo types
// AUTO - Autotuner path which will be used in future release
// GEMM - GEMM and im2row convolution path
// WINOGRAD - Winograd path which will fall back to im2row + GEMM for non compatible sizes
// DIRECT1 : Direct convolution with inputs and filters in blocked memory format
// DIRECT2 : Direct convolution with only filters in blocked memory format
// CK : Composable kernel path for convolution
enum zenConvAlgoType {
    AUTO = 0,
    GEMM = 1,
    WINOGRAD = 2,
    DIRECT1 = 3,
    DIRECT2 = 4,
    CK = 5
};

enum zenEBAlgoType {
    EB_OP_FBGEMM=0,
    EB_OP_ZENDNN=1,
};

enum zenEBThreadType {
    AUTO_ALGO = 0,
    BATCH_THREADED = 1,
    TABLE_THREADED = 2,
    HYBRID_THREADED = 3,
    CCD_THREADED = 4,
};

//class to read environment variables for zendnnn
//In future this will be used with operator memory desc
class zendnnEnv {
  public:
    uint    omp_num_threads;
    uint    zen_num_threads;
    uint    zenGEMMalgo;
    uint    zenBF16GEMMalgo;
    uint    zenINT8GEMMalgo;
    uint    zenConvAlgo;
    uint    zenEnableMemPool;
    uint    zenLibMemPoolEnable;
    uint    zenEnableTFOpts;
    uint    zenEBThreadAlgo;
    uint    zenEBAlgo;
    bool    zenINT8format;
    bool    zenWeightCache;
  private:
    //initializing ZenDNNEnv values.
    zendnnEnv() {
        omp_num_threads = zendnn_getenv_int("OMP_NUM_THREADS", 1);
        zen_num_threads = zendnn_getenv_int("ZEN_NUM_THREADS", 1);

        //ZENDNN_MATMUL_ALGO=FP32: is to enable specific FP32 MATMUL ALGO.
        //Currently ZenDNN support three ALGO path for GEMM execution
        // If value is set to 0, library decide the optimal path
        // based on the matrix sizes and other parameter settings. However,
        // this can be overridden with specific path.
        // 1. DIRECT BLIS: MatMul is redirected to BLIS GEMM directly (zenGEMMalgo=zenMatMulAlgoType::MATMUL_BLIS_GEMM1)
        // 2. ZenDNN+BLIS (zenGEMMalgo=zenMatMulAlgoType::MATMUL_BLIS_GEMM2)
        //      Case 1:
        //              ZenDNN take care of problem division and thread parallelism
        //              BLIS is used for single thread GEMM execution
        //      Case 2:
        //              MatMul is redirected to BLIS directly
        // 3. ZenDNN_sgemm: zendnn_sgemm jit based kernel (zenGEMMalgo=zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1) (current default)
        zenGEMMalgo = zendnnGetMatMulAlgo("FP32");
        if (zenGEMMalgo>zenMatMulAlgoType::MATMUL_BLIS_BLOCKED_GEMM1) {
            zenGEMMalgo = zenMatMulAlgoType::MATMUL_ZENDNN_GEMM2;
        }

        //ZENDNN_MATMUL_ALGO=BF16: is to enable specific BF16 MATMUL algo.
        //Currently ZenDNN support three ALGO path for GEMM execution
        // 0. AutoTuner, library decide the optimal path
        // based on the matrix sizes and other parameter settings. However,
        // this can be overridden with specific path.
        // 1. AOCL GEMM (zenBF16GEMMalgo=zenBF16MatMulAlgoType::MATMUL_AOCL_GEMM)
        // 2. BLOCKED JIT : MatMul is redirected to JIT (BRGEMM) (zenBF16GEMMalgo=zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT)
        // 3. JIT : MatMul is redirected to JIT (zenBF16GEMMalgo=zenBF16MatMulAlgoType::MATMUL_JIT)
        // 4. AOCL GEMM - Parallel Implementation (zenBF16GEMMalgo=zenBF16MatMulAlgoType::MATMUL_AOCL_GEMM_PAR)
        // 5. BLOCKED JIT Parallel - MatMul is redirected to JIT (BRGEMM) (zenBF16GEMMalgo=zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT_PAR)
        // 6. JIT Parallel - MatMul is redirected to JIT (zenBF16GEMMalgo=zenBF16MatMulAlgoType::MATMUL_JIT_PAR)

        zenBF16GEMMalgo = zendnnGetMatMulAlgo("BF16");
        if (zenBF16GEMMalgo>zenBF16MatMulAlgoType::MATMUL_JIT_PAR) {
            zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_JIT_PAR;
        }
        //ZENDNN_MATMUL_ALGO=INT8: is to enable specific INT8 MATMUL algo.
        //Currently ZenDNN support three ALGO path for GEMM execution
        // 0. AutoTuner, library decide the optimal path
        // based on the matrix sizes and other parameter settings. However,
        // this can be overridden with specific path.
        // 1. AOCL GEMM (zenBF16GEMMalgo=zenBF16MatMulAlgoType::MATMUL_AOCL_GEMM_INT8)
        // 2. BLOCKED JIT : MatMul is redirected to JIT (BRGEMM) (zenBF16GEMMalgo=zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT_INT8)
        // 3. JIT : MatMul is redirected to JIT (zenBF16GEMMalgo=zenBF16MatMulAlgoType::MATMUL_JIT_INT8)

        zenINT8GEMMalgo = zendnnGetMatMulAlgo("INT8");
        if (zenINT8GEMMalgo>zenINT8MatMulAlgoType::MATMUL_JIT_INT8) {
            zenINT8GEMMalgo = zenINT8MatMulAlgoType::MATMUL_JIT_INT8;
        }
        //TODO: change ZENDNN_ENABLE_MEMPOOL to ZENDNN_ENABLE_TF_MEMPOOL
        //use ZENDNN_ENABLE_ONNX_MEMPOOL for ONNX
        //Possible values for ZENDNN_ENABLE_MEMPOOL
        // 0 (GAM-TPA disable)
        // 1 (Graph level Memory Reuse)
        // 2 (Node level Memory Reuse)
        zenEnableMemPool = zendnn_getenv_int("ZENDNN_ENABLE_MEMPOOL", 1);
        if (zenEnableMemPool > 3) {
            zenEnableMemPool = 1;
        }
        zenEnableTFOpts = zendnn_getenv_int("TF_ENABLE_ZENDNN_OPTS", 1);
        //TODO: Unified FWK and LIB mempool for next release
        zenLibMemPoolEnable = zendnn_getenv_int("ZENDNN_ENABLE_MEMPOOL", 1);
        //Enabling different threading implementation.
        zenEBThreadAlgo = zendnn_getenv_int("ZENDNN_EB_THREAD_TYPE",
                                            zenEBThreadType::TABLE_THREADED);
        if (zenEBThreadAlgo>zenEBThreadType::CCD_THREADED ||
                zenEBThreadAlgo<zenEBThreadType::AUTO_ALGO) {
            zenEBThreadAlgo = zenEBThreadType::TABLE_THREADED;
        }
        zenEBAlgo = zendnn_getenv_int("ZENDNN_EB_ALGO",
                                      zenEBAlgoType::EB_OP_ZENDNN);
        if (zenEBAlgo>zenEBAlgoType::EB_OP_ZENDNN ||
                zenEBAlgo<zenEBAlgoType::EB_OP_FBGEMM) {
            zenEBAlgo = zenEBAlgoType::EB_OP_ZENDNN;
        }

        //ZENDNN_WEIGHT_CACHING is to enable/disable weight caching in MatMul
        zenWeightCache = (bool)zendnn_getenv_int("ZENDNN_WEIGHT_CACHING", 0);
        //ZENDNN_INT8_SUPPORT is to enable/disable INT8 support
        zenINT8format = (bool)zendnn_getenv_int("ZENDNN_INT8_SUPPORT", 0);
        zenConvAlgo = zendnn_getenv_int("ZENDNN_CONV_ALGO",0);
        if (zenConvAlgo <= zenConvAlgoType::AUTO ||
                zenConvAlgo > zenConvAlgoType::DIRECT2) {
            zenConvAlgo = zenConvAlgoType::GEMM;
        }
    }

    static int zenMatMulDefaultAlgo(const std::string &name) {
        if (name == "FP32") {
            return zenMatMulAlgoType::MATMUL_ZENDNN_GEMM2;
        }
        else if (name == "BF16") {
            return zenBF16MatMulAlgoType::MATMUL_JIT;
        }
        else {
            return -1;
        }
    }

    static inline int zendnnGetMatMulAlgo(const std::string &name) {
#ifdef _WIN32
        size_t sz = 0;
        static char *algoCstr;
        _dupenv_s(&algoCstr, &sz, "ZENDNN_MATMUL_ALGO");
#else
        static char *algoCstr = std::getenv("ZENDNN_MATMUL_ALGO");
#endif
        if (!algoCstr) {
            return zenMatMulDefaultAlgo(name);
        }
        std::string algoStr(algoCstr);

        size_t pos, epos;

        std::string namePlusColon(name + ":");
        pos = algoStr.find(namePlusColon);
        if (pos == std::string::npos) {
            return zenMatMulDefaultAlgo(name);
        }

        epos = pos+ namePlusColon.size();
        long x;
        char *ep;
        if (epos >= algoStr.size()) {
            assert(epos == algoStr.size());
        }
        else {
            x = strtol(algoStr.c_str() + epos, &ep, 0);
            size_t fpos = ep - algoStr.c_str();
            if (fpos - epos > 0) {
                return x;
            }
        }
        return zenMatMulDefaultAlgo(name);
    }

  public:
    static const zendnnEnv &ZenDNNEnv() {
        static const zendnnEnv envObj;
        return envObj;
    }
};

// Singleton class to use data members during execution
class zendnnOpInfo {
  private:
    zendnnOpInfo() : is_brgemm(false), is_ref_gemm_bf16(false), is_log(true) {}
  public:
    //Keep tracks if brgemm kernel is required for execution
    bool is_brgemm;
    //To select the gemm_bf16 implementation
    bool is_ref_gemm_bf16;
    bool is_log;
    static zendnnOpInfo &ZenDNNOpInfo() {
        static zendnnOpInfo obj;
        return obj;
    }
};

}


zendnn::zendnnEnv readEnv();

extern "C" {

    void zenConvolution2D_u8s8s32os32(
        const uint8_t *in_layer,
        const int no_of_images,
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
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0,
        bool reluFused=false,
        float *output_scales=nullptr
    );

    void zenConvolution2D_u8s8s32os8(
        const uint8_t *in_layer,
        const int no_of_images,
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
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0,
        bool reluFused=false,
        float *output_scales=nullptr,
        const int *zero_point_dst=nullptr,
        const int scale_count=1
    );

    void zenConvolution2D_s8s8s32os32(
        const int8_t *in_layer,
        const int no_of_images,
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
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0,
        bool reluFused=false,
        int elementwiseType = 1,
        float *output_scales=nullptr
    );

    void zenConvolution2D_s8s8s32os8(
        const int8_t *in_layer,
        const int no_of_images,
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
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0,
        bool reluFused=false,
        int elementwiseType = 1,
        float *output_scales=nullptr,
        const int *zero_point_dst=nullptr,
        const int scale_count=1
    );

    void zenConvolution2D_s8s8s16os16(
        const int8_t *in_layer,
        const int no_of_images,
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
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0,
        bool reluFused=false,
        int elementwiseType = 1,
        float *output_scales=nullptr
    );

    void zenConvolution2D_s8s8s16os8(
        const int8_t *in_layer,
        const int no_of_images,
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
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0,
        bool reluFused=false,
        int elementwiseType = 1,
        float *output_scales=nullptr,
        const int *zero_point_dst=nullptr,
        const int scale_count=1
    );

    void zenConvolution2D_u8s8s16os16(
        const uint8_t *in_layer,
        const int no_of_images,
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
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0,
        bool reluFused=false,
        float *output_scales=nullptr
    );

    void zenConvolution2D_u8s8s16os8(
        const uint8_t *in_layer,
        const int no_of_images,
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
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0,
        bool reluFused=false,
        float *output_scales=nullptr,
        const int *zero_point_dst=nullptr,
        const int scale_count=1
    );

    void zenConvolution2D_u8s8s16ou8(
        const uint8_t *in_layer,
        const int no_of_images,
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
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0,
        bool reluFused=false,
        float *output_scales=nullptr,
        const int *zero_point_dst=nullptr,
        const int scale_count=1
    );

    void zenConvolution2D_bf16bf16f32of32(
        const int16_t *in_layer,
        const int no_of_images,
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
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0,
        bool reluFused=false,
        float *output_scales=nullptr
    );

    void zenConvolution2D_bf16bf16f32obf16(
        const int16_t *in_layer,
        const int no_of_images,
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
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0,
        bool reluFused=false,
        float *output_scales=nullptr,
        const int *zero_point_dst=nullptr,
        const int scale_count=1
    );

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
        int *group_size,
        const float **Add_Array,
        int *add_shape = NULL,
        float mul_node = 1.0f,
        int batch_size = 1,
        const float **bias = NULL,
        const bool relu = 0,
        const int gelu = 0
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
        const float alpha = 1.0f,
        const float *offset = NULL,
        const float  *mean = NULL,
        const int batch_size = 1,
        const float leaky_alpha = 0.0f
    );

    void zenClipOp(
        zendnn::zendnnEnv zenEnvObj,
        float *out_layer,
        float upperbound,
        unsigned long size
    );

    void zenMatmulSplit(
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
        const bool relu,
        const int gelu,
        const float beta,
        float *output,
        const int ldc,
        bool is_weights_const
    );

    zendnn_status_t zendnn_sgemm(char transa, char transb, int64_t M, int64_t N,
                                 int64_t K, float alpha, const float *A, int64_t lda, const float *B,
                                 const int64_t ldb, float beta, float *C, int64_t ldc);

    void zenMatMul_gemm(
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
        const bool relu,
        const int gelu,
        const float beta,
        float *output,
        const int ldc,
        bool is_weights_const
    );

    int auto_compute_matmul(
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
        const bool relu,
        const int gelu,
        const float beta,
        float *output,
        const int ldc,
        bool is_weights_const
    );
    int auto_compute_conv(
        int supportedPath,
        void *in_layer,
        int no_of_images,
        int channels,
        int height,
        int width,
        int8_t *filter,
        int no_of_filter,
        int kernel_h,
        int kernel_w,
        int pad_t,
        int pad_l,
        int pad_b,
        int pad_r,
        int stride_h,
        int stride_w,
        void *bias,
        void *out_layer,
        int out_height,
        int out_width,
        bool concat,
        int filter_offset,
        int total_filters,
        bool reluFused,
        int elementwiseType,
        float *output_scales,
        const int *zero_point_dst,
        int scale_count
    );
    void zendnnConvolutionLPGEMM(
        int supportedPath,
        int zendnn_lpgemm_algo,
        void *src,
        int no_of_images,
        int channels,
        int height,
        int width,
        int8_t *filter,
        int no_of_filter,
        int kernel_h,
        int kernel_w,
        int pad_t,
        int pad_l,
        int pad_b,
        int pad_r,
        int stride_h,
        int stride_w,
        void *bias,
        void *dst,
        int out_height,
        int out_width,
        bool concat,
        int filter_offset,
        int total_filters,
        bool reluFused,
        int elementwiseType,
        float *output_scales,
        const int *zero_point_dst,
        int scale_size
    );
}
