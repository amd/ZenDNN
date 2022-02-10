/*******************************************************************************
* Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <omp.h>
#include <sys/sysinfo.h>
#include <cblas.h>
#include <time.h>
#include <sys/time.h>
#include "zendnn_convolution_winograd.hpp"
#include "zendnn_private.hpp"
#include "zendnn_logging.hpp"
#include "zendnn_helper.hpp"

using namespace zendnn;

//BLIS_SMALL_MATRIX is depend on the L3 shared cache
//with 24C ROME, 3 cores shares one L3 cache and with 64C ROME,
//4 cores shares one L3 cache.
//TODO: Read cache info from underlying platform and decide this value.
#define BLIS_SMALL_MATRIX       392 //Based on network layer i/p
#define BLIS_SMALL_MATRIX_MILAN 784 //Based on network layer i/p
#define CONV_INPUT_SIZE         7168 //Based on heuristic with googlenet,resnet and vgg
#define CONV_INPUT_HEIGHT       80 //Based on heuristic with googlenet,resnet and vgg. After 80 transformation function degrades the performance
#define SMALL_CONV_INPUT        10 //Based on heuristic with googlenet,resnet and vgg. After 10 transformation function degrades the performance
#define SPLIT_CONV_INPUT        20


#define DIRECT_CONV_GEMV        0
#define WINOGRAD_CONV           1



//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner for im2row, gemm and bias operation
//internally these operations runs sequntially....we call it threading outside operation
//TODO: Add inplace concat support
void zenConvolution2Dbase(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int images,
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
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

#if 0
    unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) {
        thread_qty = 1;
    }

    struct timeval start, end;
    gettimeofday(&start, 0);

#endif
    int batchsize = images;
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2Dbase, no_of_images=", batchsize,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w);

    unsigned int blis_num_threads = zendnn_getenv_int("BLIS_NUM_THREADS");
    if (blis_num_threads == 0) {
        blis_num_threads = 1;
    }

    unsigned int thread_qty = zenEnvObj.omp_num_threads/blis_num_threads;
    //Need to change this for latency optimization
    if (thread_qty > images) {
        thread_qty = images;
    }

    unsigned long data_col_size = ((unsigned long)(kernel_h*kernel_w*channels)*
                                   (out_height*out_width)*sizeof(float)*thread_qty);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = (float *)aligned_alloc(ALIGNED_OFFSET, data_col_size);

    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2Dbase Memory Error while allocating patch matrix");
        return;
    }

    #pragma omp parallel num_threads(thread_qty)
    {
        unsigned int loopCount = (images%thread_qty)==0 ? images/thread_qty :
                                 (images/thread_qty)+1;
        for (int i=0; i<loopCount; i++) {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            if (threadOffset >= images) {
                break;
            }
            unsigned long inputOffset = ((unsigned long)channels*height*width*threadOffset);
            unsigned long patchInputOffset = ((unsigned long)(kernel_h*kernel_w*channels)*
                                              (out_height*out_width) * omp_get_thread_num());
            unsigned long outputOffset = ((unsigned long)no_of_filter*
                                          (out_height*out_width)* threadOffset);

            //im2row is more efficient than im2col with NHWC
            //im2rowNHWC(in_layer+inputOffset, channels, height, width, kernel_h, kernel_w,
            //      pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, data_col + patchInputOffset);
            im2rowNHWCsplit(in_layer+inputOffset, channels, height, width, kernel_h,
                            kernel_w,
                            pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, data_col + patchInputOffset,
                            out_height, 0, blis_num_threads);

            //AMD BLIS bases matrix multiplication
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out_height*out_width,
                        no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                        data_col+patchInputOffset, channels*kernel_h*kernel_w, filter, no_of_filter,
                        0.0f, out_layer+outputOffset, no_of_filter);

            unsigned long biasOffset = outputOffset;
            zenPostOps(zenEnvObj, out_layer, elementwise_input,out_height, out_width,
                       no_of_filter, no_of_filter,
                       biasOffset, bias,
                       relu, false, scale, blis_num_threads);
        }
    }
    free(data_col);
#if 0
    gettimeofday(&end, 0);
    float elapsed;
    elapsed = timedifference_msec(start, end);
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2D_best, no_of_images=", no_of_images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_h=", pad_h, " pad_w=", pad_w,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " Time=", elapsed, "ms");
#endif

}

//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner for im2row and bias operation
//for gemm BLIS will take care of the parallelism.
//This is Memory optimized as this works with smaller matrix chunks limiting memory requirment to no. of threads
//and do the work in no_images/no_of_threads chunks
//We call it mix of threading inside and outside conv operation
//TODO: Add inplace concat support
void zenConvolution2DbaseVer5(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int images,
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
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

    int batchsize = images;
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2D_ver5, no_of_images=", batchsize,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w);

    unsigned int blis_num_threads = zendnn_getenv_int("BLIS_NUM_THREADS");
    if (blis_num_threads == 0) {
        blis_num_threads = 1;
    }

    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    //Need to change this for latency optimization
    if (thread_qty > images) {
        thread_qty = images;
    }

    unsigned long data_col_size = ((unsigned long)(kernel_h*kernel_w*channels)*
                                   (out_height*out_width)*sizeof(float)*thread_qty);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = (float *)aligned_alloc(ALIGNED_OFFSET, data_col_size);

    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2Dbase Memory Error while allocating patch matrix");
        return;
    }

    unsigned int loopCount = (images%thread_qty)==0 ? images/thread_qty :
                             (images/thread_qty)+1;

    for (int i=0; i<loopCount; i++) {

        unsigned int outBatchSize = thread_qty;
        if (i==(loopCount-1) && (images%thread_qty)!=0) {
            outBatchSize = images - ((images/thread_qty) * thread_qty);
        }

        unsigned long outputOffset = (no_of_filter* (out_height*out_width)* thread_qty *
                                      i);
        #pragma omp parallel num_threads(thread_qty)
        {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            if (threadOffset < images) {

                unsigned long patchInputOffset = ((unsigned long)(kernel_h*kernel_w*channels)*
                                                  (out_height*out_width) * omp_get_thread_num());
                unsigned long inputOffset = ((unsigned long)channels*height*width*threadOffset);

                im2rowNHWCsplit(in_layer+inputOffset, channels, height, width, kernel_h,
                                kernel_w,
                                pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, data_col + patchInputOffset,
                                out_height, 0, 1);
            }
        }

        #pragma omp parallel for num_threads(thread_qty/blis_num_threads)
        for (int j=0; j<outBatchSize; j++) {
            unsigned long patchInputOffset = (((unsigned long)kernel_h*kernel_w*channels)*
                                              (out_height*out_width) * j);
            unsigned long outputOffset2 = ((unsigned long)no_of_filter*
                                           (out_height*out_width)* j);
            //AMD BLIS bases matrix multiplication
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out_height*out_width,
                        no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                        data_col+patchInputOffset, channels*kernel_h*kernel_w, filter, no_of_filter,
                        0.0f, out_layer+outputOffset+outputOffset2, no_of_filter);

        }
        unsigned long biasOffset = outputOffset;
        zenPostOps(zenEnvObj, out_layer, elementwise_input,out_height,
                   out_width*outBatchSize,
                   no_of_filter, no_of_filter,
                   biasOffset, bias, relu, false, scale, thread_qty);

    }
    free(data_col);
}


//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner for im2row, gemm and bias operation
//internally these operations runs sequntially....we call it threading outside operation
//TODO: Add inplace concat support
void zenConvolution2DsmallGemm(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int images,
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
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

#if 0
    unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) {
        thread_qty = 1;
    }

    struct timeval start, end;
    gettimeofday(&start, 0);
#endif
    unsigned int thread_qty = zenEnvObj.omp_num_threads;

    //Need to change this for latency optimization
    if (thread_qty > images) {
        thread_qty = images;
    }

    unsigned long data_col_size = ((unsigned long)(kernel_h*kernel_w*channels)*
                                   (out_height*out_width)*sizeof(float)*thread_qty);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = (float *)aligned_alloc(ALIGNED_OFFSET, data_col_size);


    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DsmallGemm Memory Error while allocating patch matrix");
        return;
    }

    int height_col =
        out_height; //(height + pad_h + pad_w - kernel_h) / stride_h + 1;
    int width_col = out_width;//(width + pad_h + pad_w - kernel_w) / stride_w + 1;
    #pragma omp parallel num_threads(thread_qty)
    {
        unsigned int loopCount = (images%thread_qty)==0 ? images/thread_qty :
                                 (images/thread_qty)+1;
        for (int i=0; i<loopCount; i++) {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            if (threadOffset >= images) {
                break;
            }
            unsigned long inputOffset = ((unsigned long)channels*height*width*threadOffset);
            unsigned long patchInputOffset = ((unsigned long)(kernel_h*kernel_w*channels)*
                                              (out_height*out_width) * omp_get_thread_num());
            unsigned long outputOffset = ((unsigned long)no_of_filter*
                                          (out_height*out_width)*
                                          threadOffset);

            int gemmRows = BLIS_SMALL_MATRIX/width_col;
            int gemmRowsLast = (height_col%gemmRows)==0? gemmRows : (height_col%gemmRows);
            int height_colLoop = (height_col%gemmRows)==0? (height_col/gemmRows) :
                                 (height_col/gemmRows)+1;

            for (int k=0; k<height_colLoop; k++) {
                unsigned long patchHeightOffset = (unsigned long)k*gemmRows*width_col*
                                                  (kernel_h*kernel_w*channels);

                if (k==(height_colLoop-1)) {
                    im2rowNHWCsplit(in_layer+inputOffset, channels, height, width, kernel_h,
                                    kernel_w, pad_t, pad_l, pad_b, pad_r,
                                    stride_h, stride_w, data_col + patchInputOffset + patchHeightOffset,
                                    gemmRowsLast, gemmRows*k, 1);


                    //AMD BLIS bases matrix multiplication
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, width_col*gemmRowsLast,
                                no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                                data_col+patchInputOffset+patchHeightOffset, channels*kernel_h*kernel_w, filter,
                                no_of_filter, 0.0f,
                                out_layer+outputOffset+(width_col*no_of_filter*gemmRows*k), no_of_filter);

                    unsigned long biasOffset = outputOffset+(width_col*no_of_filter*gemmRows*k);
                    zenPostOps(zenEnvObj, out_layer, elementwise_input,width_col, gemmRowsLast,
                               no_of_filter, no_of_filter,
                               biasOffset, bias,
                               relu, false, scale, 1);
                }
                else {
                    im2rowNHWCsplit(in_layer+inputOffset, channels, height, width, kernel_h,
                                    kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w,
                                    data_col + patchInputOffset + patchHeightOffset, gemmRows, gemmRows*k, 1);

                    //AMD BLIS bases matrix multiplication
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, width_col*gemmRows,
                                no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                                data_col+patchInputOffset+patchHeightOffset, channels*kernel_h*kernel_w, filter,
                                no_of_filter, 0.0f,
                                out_layer+outputOffset+(width_col*no_of_filter*gemmRows*k), no_of_filter);

                    unsigned long biasOffset = outputOffset+(width_col*no_of_filter*gemmRows*k);
                    zenPostOps(zenEnvObj, out_layer, elementwise_input,width_col, gemmRowsLast,
                               no_of_filter, no_of_filter,
                               biasOffset, bias,
                               relu, false, scale, 1);
                }

            }
        }
    }
    free(data_col);
#if 0
    gettimeofday(&end, 0);
    float elapsed;
    elapsed = timedifference_msec(start, end);
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2D_bestSmallGemm, images=", images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_h=", pad_h, " pad_w=", pad_w,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " Time=", elapsed, "ms");
#endif

}



//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner for im2row, gemm and bias operation
//internally these operations runs sequntially....we call it threading outside operation
void zenConvolution2DsmallGemmVer2(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int images,
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
    const bool sum_fused,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

#if 0
    unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) {
        thread_qty = 1;
    }

    struct timeval start, end;
    gettimeofday(&start, 0);
#endif
    int batchsize = images;
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2DsmallGemmVer2, no_of_images=",
               batchsize,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " isConcat=", concat, " filter_offset=", filter_offset,
               " total_filters=", total_filters);

    float gemm_beta = 0.0;
    if (sum_fused) {
        gemm_beta = 1.0;
    }

    unsigned int thread_qty = zenEnvObj.omp_num_threads;

    int height_col =
        out_height;//(height + pad_h + pad_w - kernel_h) / stride_h + 1;
    int width_col = out_width;//(width + pad_h + pad_w - kernel_w) / stride_w + 1;

    int blis_num_threads = 1;

#if BLIS_EXPERT
    //Enable Nested parallelism when patch matrix outer loop is not able to use all threads
    //TODO: Need to try with Forced nested parallelism for all sizes, try with various blis_num_threads values
    if (thread_qty > images) {
        blis_num_threads = (thread_qty%images)==0?(thread_qty/images):((
                               thread_qty/images)+1);
    }
    else {
        blis_num_threads = thread_qty<1?thread_qty:1;
    }
    thread_qty = (thread_qty%blis_num_threads)==0?(thread_qty/blis_num_threads):
                 (thread_qty/blis_num_threads)+1;
    omp_set_max_active_levels(2);
#else
    if (thread_qty > images) {
        thread_qty = images;
    }
    omp_set_max_active_levels(1);
#endif

    //unsigned int no_of_images = images;
    unsigned long data_col_size = ((unsigned long)(kernel_h*kernel_w*channels)*
                                   (out_height*out_width)*sizeof(float)*thread_qty);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = NULL;
    int zenLibPoolEnable = zenEnvObj.zenLibMemPoolEnable;
    ZenLibMemoryPool *zenLibPoolBuffer;

    if ((kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
            out_width == width)) {
        data_col = (float *)in_layer;
    }
    else {

        //ZenLibMemPool Optimization reuse tmp buffers from the pool. By default
        //  its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
        //  pool optimization
        //  Cases where buffers in pool are not free or requested size is more
        //  than available buffer size in Pool, control will fall back to
        //  default way of allocation
        if (zenLibPoolEnable) {
            zenLibPoolBuffer = ZenLibMemoryPool::getZenLibMemPool(0);
            if (zenLibPoolBuffer) {
                int status = zenLibPoolBuffer->acquireZenLibPoolBuf(&data_col, data_col_size,
                             1);
                if (status) {
                    zenLibPoolEnable = false;
                }
            }
            else {
                zenLibPoolEnable = false;
            }
        }
        if (!zenLibPoolEnable) {
            data_col = (float *)aligned_alloc(ALIGNED_OFFSET, data_col_size);
        }

    }
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DsmallGemmVer2 Memory Error while allocating patch matrix");
        return;
    }

    unsigned int ldc = no_of_filter;
    if (concat) {
        ldc = total_filters;
    }

    #pragma omp parallel num_threads(thread_qty)
    {

#if BLIS_EXPERT
        if ((thread_qty%blis_num_threads)!=0 && omp_get_num_threads()==(thread_qty-1)) {
            blis_num_threads = thread_qty%blis_num_threads;
        }
        //creating blis expert interface
        blis_expert blis_obj(blis_num_threads, BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE);
        //bli_rntm_set_ways(1, 1, blis_num_threads, 1, 1, &blis_obj.rntm);
        bli_setsc(gemm_beta, 0.0, &blis_obj.beta);
#endif
        unsigned int loopCount = (images%thread_qty)==0 ? images/thread_qty :
                                 (images/thread_qty)+1;
        for (int i=0; i<loopCount; i++) {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            if (threadOffset >= images) {
                break;
            }

            unsigned long inputOffset = ((unsigned long)channels*height*width*threadOffset);
            unsigned long patchInputOffset = ((unsigned long)(kernel_h*kernel_w*channels)*
                                              (out_height*out_width) * omp_get_thread_num());

            if ((kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
                    out_width == width)) {
                patchInputOffset = inputOffset;
            }

            unsigned long outputOffset = ((unsigned long)ldc*
                                          (out_height*out_width)* threadOffset);

            int gemmRows = BLIS_SMALL_MATRIX/width_col;
            int gemmRowsLast = (height_col%gemmRows)==0? gemmRows : (height_col%gemmRows);
            int height_colLoop = (height_col%gemmRows)==0? (height_col/gemmRows) :
                                 (height_col/gemmRows)+1;

            for (int k=0; k<height_colLoop; k++) {
                //im2row is more efficient than im2col with NHWC
                unsigned long patchHeightOffset = (unsigned long)k*gemmRows*width_col*
                                                  (kernel_h*kernel_w*channels);
                if (k==(height_colLoop-1)) {
                    if (!(kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
                            out_width == width))
                        im2rowNHWCsplit(in_layer+inputOffset, channels, height, width, kernel_h,
                                        kernel_w, pad_t, pad_l, pad_b, pad_r,
                                        stride_h, stride_w, data_col + patchInputOffset + patchHeightOffset,
                                        gemmRowsLast, gemmRows*k, blis_num_threads);
                    unsigned long offset = ((unsigned long)width_col*ldc*gemmRows*k) +
                                           filter_offset;
#if BLIS_EXPERT
                    bli_obj_create_with_attached_buffer(blis_obj.dt, width_col*gemmRowsLast,
                                                        channels*kernel_h*kernel_w,
                                                        data_col+patchInputOffset+patchHeightOffset, channels*kernel_h*kernel_w, 1,
                                                        &blis_obj.a);
                    bli_obj_create_with_attached_buffer(blis_obj.dt, channels*kernel_h*kernel_w,
                                                        no_of_filter,
                                                        (void *)filter, no_of_filter, 1, &blis_obj.b);
                    bli_obj_create_with_attached_buffer(blis_obj.dt, width_col*gemmRowsLast,
                                                        no_of_filter,
                                                        out_layer+outputOffset+offset, ldc, 1, &blis_obj.c);
                    bli_gemm_ex(&blis_obj.alpha, &blis_obj.a, &blis_obj.b, &blis_obj.beta,
                                &blis_obj.c, NULL, &blis_obj.rntm);
#else
                    //AMD BLIS bases matrix multiplication
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, width_col*gemmRowsLast,
                                no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                                data_col+patchInputOffset+patchHeightOffset, channels*kernel_h*kernel_w, filter,
                                no_of_filter, gemm_beta,
                                out_layer+outputOffset+offset, ldc);
#endif
                    //Below Bias and activation code can be eliminated if not required
                    unsigned long biasOffset = outputOffset+offset;
                    zenPostOps(zenEnvObj, out_layer, elementwise_input,width_col, gemmRowsLast,
                               no_of_filter, ldc,
                               biasOffset, bias,
                               relu, false, scale, blis_num_threads);
                }
                else {
                    if (!(kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
                            out_width == width))
                        im2rowNHWCsplit(in_layer+inputOffset, channels, height, width, kernel_h,
                                        kernel_w, pad_t, pad_l, pad_b, pad_r,
                                        stride_h, stride_w, data_col + patchInputOffset + patchHeightOffset, gemmRows,
                                        gemmRows*k, blis_num_threads);
                    unsigned long offset = ((unsigned long)width_col*ldc*gemmRows*k) +
                                           filter_offset;
#if BLIS_EXPERT
                    bli_obj_create_with_attached_buffer(blis_obj.dt, width_col*gemmRows,
                                                        channels*kernel_h*kernel_w,
                                                        data_col+patchInputOffset+patchHeightOffset, channels*kernel_h*kernel_w, 1,
                                                        &blis_obj.a);
                    bli_obj_create_with_attached_buffer(blis_obj.dt, channels*kernel_h*kernel_w,
                                                        no_of_filter,
                                                        (void *)filter, no_of_filter, 1, &blis_obj.b);
                    bli_obj_create_with_attached_buffer(blis_obj.dt, width_col*gemmRows,
                                                        no_of_filter,
                                                        out_layer+outputOffset+offset, ldc, 1, &blis_obj.c);
                    bli_gemm_ex(&blis_obj.alpha, &blis_obj.a, &blis_obj.b, &blis_obj.beta,
                                &blis_obj.c, NULL, &blis_obj.rntm);
#else
                    //AMD BLIS bases matrix multiplication
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, width_col*gemmRows,
                                no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                                data_col+patchInputOffset+patchHeightOffset, channels*kernel_h*kernel_w, filter,
                                no_of_filter, gemm_beta,
                                out_layer+outputOffset+offset, ldc);
#endif
                    unsigned long biasOffset = outputOffset+offset;
                    zenPostOps(zenEnvObj, out_layer, elementwise_input,width_col, gemmRows,
                               no_of_filter, ldc,
                               biasOffset, bias, relu, false,
                               scale, blis_num_threads);
                }

            }
        }
    }
    if (!(kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
            out_width == width)) {
        //If ZenMemPool Optimization is enabled(default), update the state of
        //  Memory pool based on input_array address
        if (zenLibPoolEnable) {
            zenLibPoolBuffer->zenLibMemPoolFree((float *)data_col);
        }
        else {
            free(data_col);
        }
    }

#if 0
    gettimeofday(&end, 0);
    float elapsed;
    elapsed = timedifference_msec(start, end);
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2D_bestSmallGemmVer2, no_of_images=",
               images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_h=", pad_h, " pad_w=", pad_w,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " Time=", elapsed, "ms");
#endif

}


//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
void zenConvolution2DGemm1x1Direct(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int images,
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
    const bool sum_fused,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

#if 0
    unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) {
        thread_qty = 1;
    }

    struct timeval start, end;
    gettimeofday(&start, 0);
#endif
    int batchsize = images;
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2DGemm1x1Direct, no_of_images=",
               batchsize,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " isConcat=", concat, " filter_offset=", filter_offset,
               " total_filters=", total_filters);

    float gemm_beta = 0.0;
    if (sum_fused) {
        gemm_beta = 1.0;
    }

    int blis_num_threads = 1;
    unsigned int thread_qty = zenEnvObj.omp_num_threads;

#if BLIS_EXPERT
    //2 performs optimal for nested parallelism with BLIS
    //Assuming OMP_NUM_THREADS will always multiple of 2
    if (images > 1) {
        if (thread_qty > images) {
            blis_num_threads = (thread_qty%images)==0?(thread_qty/images):((
                                   thread_qty/images)+1);
        }
        else {
            blis_num_threads = thread_qty<2?thread_qty:2;
        }
    }
    else {
        //For BS=1, we use all threads for GEMM parallelism
        blis_num_threads = thread_qty;
    }
    thread_qty = (thread_qty%blis_num_threads)==0?(thread_qty/blis_num_threads):
                 (thread_qty/blis_num_threads)+1;
    omp_set_max_active_levels(2);
#else
    if (thread_qty > images) {
        thread_qty = images;
    }
    omp_set_max_active_levels(1);
#endif

    int height_col =
        out_height;//(height + pad_h + pad_w - kernel_h) / stride_h + 1;
    int width_col = out_width;//(width + pad_h + pad_w - kernel_w) / stride_w + 1;

    unsigned int image_merge_count = images;
    unsigned long gemmRows = ((unsigned long)width_col* height_col *
                              image_merge_count);
    int gemmRowsCount = gemmRows/thread_qty;

    unsigned int ldc = no_of_filter;
    if (concat) {
        ldc = total_filters;
    }

    #pragma omp parallel num_threads(thread_qty) private(gemmRows)
    {
        gemmRows = ((unsigned long)width_col* height_col * image_merge_count);
        if (omp_get_thread_num()==(thread_qty-1)) {
            gemmRows = gemmRowsCount + (gemmRows%thread_qty);
        }
        else {
            gemmRows = gemmRowsCount;
        }

        unsigned long inputOffset = ((unsigned long)
                                     gemmRowsCount*channels*kernel_h*kernel_w*omp_get_thread_num());
        unsigned long outputOffset = ((unsigned long)
                                      gemmRowsCount*ldc*omp_get_thread_num());

        unsigned int offset = filter_offset;
#if BLIS_EXPERT
        if ((thread_qty%blis_num_threads)!=0 && omp_get_num_threads()==(thread_qty-1)) {
            blis_num_threads = thread_qty%blis_num_threads;
        }
        //creating blis expert interface
        blis_expert blis_obj(blis_num_threads, BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE);
        //bli_rntm_set_ways(1, 1, blis_num_threads, 1, 1, &blis_obj.rntm);
        bli_setsc(gemm_beta, 0.0, &blis_obj.beta);

        bli_obj_create_with_attached_buffer(blis_obj.dt, gemmRows,
                                            channels*kernel_h*kernel_w,
                                            (float *)in_layer+inputOffset, channels*kernel_h*kernel_w, 1, &blis_obj.a);
        bli_obj_create_with_attached_buffer(blis_obj.dt, channels*kernel_h*kernel_w,
                                            no_of_filter,
                                            (void *)filter, no_of_filter, 1, &blis_obj.b);
        bli_obj_create_with_attached_buffer(blis_obj.dt, gemmRows, no_of_filter,
                                            out_layer+outputOffset+offset, ldc, 1, &blis_obj.c);

        bli_gemm_ex(&blis_obj.alpha, &blis_obj.a, &blis_obj.b, &blis_obj.beta,
                    &blis_obj.c, NULL, &blis_obj.rntm);
#else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, gemmRows, no_of_filter,
                    channels*kernel_h*kernel_w, 1.0f,
                    in_layer+inputOffset, channels*kernel_h*kernel_w, filter, no_of_filter,
                    gemm_beta,
                    out_layer+outputOffset+offset, ldc);
#endif
        unsigned long biasOffset = outputOffset+offset;
        zenPostOps(zenEnvObj, out_layer, elementwise_input,gemmRows, 1, no_of_filter,
                   ldc, biasOffset,
                   bias, relu, false, scale,
                   blis_num_threads);
    }

#if 0
    gettimeofday(&end, 0);
    float elapsed;
    elapsed = timedifference_msec(start, end);
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2D_bestSmallGemmVer3, no_of_images=",
               images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_h=", pad_h, " pad_w=", pad_w,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " Time=", elapsed, "ms");
#endif

}


//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner for im2row, gemm and bias operation
void zenConvolution2DsmallGemmMerge(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int images,
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
    const bool sum_fused,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

#if 0
    unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) {
        thread_qty = 1;
    }

    struct timeval start, end;
    gettimeofday(&start, 0);
#endif
    int batchsize = images;
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2DsmallGemmMerge, no_of_images=",
               batchsize,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " isConcat=", concat, " filter_offset=", filter_offset,
               " total_filters=", total_filters);

    float gemm_beta = 0.0;
    if (sum_fused) {
        gemm_beta = 1.0;
    }

    unsigned int thread_qty = zenEnvObj.omp_num_threads;

    int height_col =
        out_height;//(height + pad_h + pad_w - kernel_h) / stride_h + 1;
    int width_col = out_width;//(width + pad_h + pad_w - kernel_w) / stride_w + 1;


    //Overriding parameters  to combine images
    unsigned int image_merge_count = 1;
    unsigned int no_of_images = images;

    //Merging image in order to make SGEMM M close to N
    image_merge_count = (no_of_filter/(width_col*height_col))==0?1:
                        (no_of_filter/(width_col*height_col));
    //Merging images beyond 4 degrades the performance
    image_merge_count = image_merge_count>=4?4:image_merge_count;
    no_of_images = (images%image_merge_count)==0 ? images/image_merge_count :
                   (images/image_merge_count)+1;

    int blis_num_threads = 1;
#if BLIS_EXPERT
    //Enable Nested parallelism when patch matrix outer loop is not able to use all threads
    //TODO: Need to try with Forced nested parallelism for all sizes, try with various blis_num_threads values
    if (thread_qty > no_of_images) {
        blis_num_threads = (thread_qty%no_of_images)==0?(thread_qty/no_of_images):((
                               thread_qty/no_of_images)+1);
    }
    else {
        blis_num_threads = thread_qty<1?thread_qty:1;
    }
    thread_qty = (thread_qty%blis_num_threads)==0?(thread_qty/blis_num_threads):
                 (thread_qty/blis_num_threads)+1;
    omp_set_max_active_levels(2);
#else
    if (thread_qty > no_of_images) {
        thread_qty = no_of_images;
    }
    omp_set_max_active_levels(1);
#endif

    unsigned long data_col_size = ((unsigned long)(kernel_h*kernel_w*channels)*
                                   (out_height*out_width)*sizeof(float)*thread_qty* image_merge_count);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = NULL;
    int zenLibPoolEnable = zenEnvObj.zenLibMemPoolEnable;
    ZenLibMemoryPool *zenLibPoolBuffer;

    if ((kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
            out_width == width)) {
        data_col = (float *)in_layer;
    }
    else {

        //ZenLibMemPool Optimization reuse tmp buffers from the pool. By default
        //  its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
        //  pool optimization
        //  Cases where buffers in pool are not free or requested size is more
        //  than available buffer size in Pool, control will fall back to
        //  default way of allocation
        if (zenLibPoolEnable) {
            zenLibPoolBuffer = ZenLibMemoryPool::getZenLibMemPool(0);
            if (zenLibPoolBuffer) {
                int status = zenLibPoolBuffer->acquireZenLibPoolBuf(&data_col, data_col_size,
                             1);
                if (status) {
                    zenLibPoolEnable = false;
                }
            }
            else {
                zenLibPoolEnable = false;
            }
        }
        if (!zenLibPoolEnable) {
            data_col = (float *)aligned_alloc(ALIGNED_OFFSET, data_col_size);
        }
    }
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DsmallGemmMerge Memory Error while allocating patch matrix");
        return;
    }

    unsigned int ldc = no_of_filter;
    if (concat) {
        ldc = total_filters;
    }

    #pragma omp parallel num_threads(thread_qty)
    {

#if BLIS_EXPERT
        if ((thread_qty%blis_num_threads)!=0 && omp_get_num_threads()==(thread_qty-1)) {
            blis_num_threads = thread_qty%blis_num_threads;
        }
        //creating blis expert interface
        blis_expert blis_obj(blis_num_threads, BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE);
        bli_setsc(gemm_beta, 0.0, &blis_obj.beta);
        //bli_rntm_set_ways(1, 1, blis_num_threads, 1, 1, &blis_obj.rntm);
#endif

        unsigned int loopCount = (no_of_images%thread_qty)==0 ?
                                 no_of_images/thread_qty : (no_of_images/thread_qty)+1;
        for (int i=0; i<loopCount; i++) {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            if (threadOffset >= no_of_images) {
                break;
            }

            unsigned long inputOffset = ((unsigned long)
                                         channels*height*width*threadOffset*image_merge_count);
            unsigned long patchInputOffset = ((unsigned long)(kernel_h*kernel_w*channels)*
                                              (out_height*out_width) * omp_get_thread_num() * image_merge_count);
            unsigned long outputOffset = ((unsigned long)ldc*
                                          (out_height*out_width)* threadOffset *image_merge_count);

            unsigned int merge_loop = image_merge_count;

            if (threadOffset == (no_of_images-1)) {
                if (images%image_merge_count != 0) {
                    merge_loop = images%image_merge_count;
                }
            }

            unsigned long gemmRows = ((unsigned long)width_col* height_col * merge_loop);
            unsigned long gemmRowsLast = gemmRows;

            unsigned int offset = filter_offset;
            if (!(kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
                    out_width == width)) {
                for (int k=0; k<merge_loop; k++) {
                    unsigned long patchHeightOffset = (unsigned long)k*(kernel_h*kernel_w*channels)
                                                      *(out_height*out_width);
                    im2rowNHWCsplit(in_layer+inputOffset+((unsigned long)k*(channels*height*width)),
                                    channels, height, width,
                                    kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w,
                                    data_col + patchInputOffset + patchHeightOffset, height_col, 0,
                                    blis_num_threads);
                }
#if BLIS_EXPERT
                bli_obj_create_with_attached_buffer(blis_obj.dt, gemmRowsLast,
                                                    channels*kernel_h*kernel_w,
                                                    data_col+patchInputOffset, channels*kernel_h*kernel_w, 1, &blis_obj.a);
                bli_obj_create_with_attached_buffer(blis_obj.dt, channels*kernel_h*kernel_w,
                                                    no_of_filter,
                                                    (void *)filter, no_of_filter, 1, &blis_obj.b);
                bli_obj_create_with_attached_buffer(blis_obj.dt, gemmRowsLast, no_of_filter,
                                                    out_layer+outputOffset+offset, ldc, 1, &blis_obj.c);

                bli_gemm_ex(&blis_obj.alpha, &blis_obj.a, &blis_obj.b, &blis_obj.beta,
                            &blis_obj.c, NULL, &blis_obj.rntm);
#else
                //AMD BLIS bases matrix multiplication
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, gemmRowsLast,
                            no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                            data_col+patchInputOffset, channels*kernel_h*kernel_w, filter, no_of_filter,
                            gemm_beta, out_layer+outputOffset+offset, ldc);
#endif
            }
            else {
#if BLIS_EXPERT
                bli_obj_create_with_attached_buffer(blis_obj.dt, gemmRowsLast,
                                                    channels*kernel_h*kernel_w,
                                                    data_col+inputOffset, channels*kernel_h*kernel_w, 1, &blis_obj.a);
                bli_obj_create_with_attached_buffer(blis_obj.dt, channels*kernel_h*kernel_w,
                                                    no_of_filter,
                                                    (void *)filter, no_of_filter, 1, &blis_obj.b);
                bli_obj_create_with_attached_buffer(blis_obj.dt, gemmRowsLast, no_of_filter,
                                                    out_layer+outputOffset+offset, ldc, 1, &blis_obj.c);

                bli_gemm_ex(&blis_obj.alpha, &blis_obj.a, &blis_obj.b, &blis_obj.beta,
                            &blis_obj.c, NULL, &blis_obj.rntm);
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, gemmRowsLast,
                            no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                            data_col+inputOffset, channels*kernel_h*kernel_w, filter, no_of_filter,
                            gemm_beta,
                            out_layer+outputOffset+offset, ldc);
#endif
            }

            unsigned long biasOffset = outputOffset+offset;
            zenPostOps(zenEnvObj, out_layer, elementwise_input,gemmRowsLast, 1,
                       no_of_filter, ldc,
                       biasOffset, bias, relu, false,
                       scale, blis_num_threads);
            continue;
        }
    }
    if (!(kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
            out_width == width)) {
        //If ZenMemPool Optimization is enabled(default), update the state of
        //  Memory pool based on input_array address
        if (zenLibPoolEnable) {
            zenLibPoolBuffer->zenLibMemPoolFree((float *)data_col);
        }
        else {
            free(data_col);
        }
    }

#if 0
    gettimeofday(&end, 0);
    float elapsed;
    elapsed = timedifference_msec(start, end);
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2D_bestSmallGemmVer2, no_of_images=",
               no_of_images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_h=", pad_h, " pad_w=", pad_w,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " Time=", elapsed, "ms");
#endif

}

//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
void zenConvolution2DsmallGemm1x1(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int images,
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
    const bool sum_fused,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

#if 0
    unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) {
        thread_qty = 1;
    }

    struct timeval start, end;
    gettimeofday(&start, 0);
#endif
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    int batchsize = images;
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2DsmallGemm1x1, no_of_images=",
               batchsize,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " isConcat=", concat, " filter_offset=", filter_offset,
               " total_filters=", total_filters);

    float gemm_beta = 0.0;
    if (sum_fused) {
        gemm_beta = 1.0;
    }

    int height_col =
        out_height;//(height + pad_h + pad_w - kernel_h) / stride_h + 1;
    int width_col = out_width;//(width + pad_h + pad_w - kernel_w) / stride_w + 1;

    int blis_num_threads = 1;
#if BLIS_EXPERT
    //2 performs optimal for nested parallelism with BLIS
    //TODO: Need to try with different values with all models
    if (thread_qty > images) {
        blis_num_threads = (thread_qty%images)==0?(thread_qty/images):((
                               thread_qty/images)+1);
    }
    else {
        blis_num_threads = thread_qty<2?thread_qty:2;
    }
    thread_qty = (thread_qty%blis_num_threads)==0?(thread_qty/blis_num_threads):
                 (thread_qty/blis_num_threads)+1;
    omp_set_max_active_levels(2);
#else
    if (thread_qty > images) {
        thread_qty = images;
    }
    omp_set_max_active_levels(1);
#endif

    float *data_col = NULL;
    data_col = (float *)in_layer;

    unsigned int ldc = no_of_filter;
    if (concat) {
        ldc = total_filters;
    }

    unsigned int image_merge_count_rem = images%thread_qty;
    #pragma omp parallel num_threads(thread_qty)
    {

#if BLIS_EXPERT
        if ((thread_qty%blis_num_threads)!=0 && omp_get_num_threads()==(thread_qty-1)) {
            blis_num_threads = thread_qty%blis_num_threads;
        }
        //creating blis expert interface
        blis_expert blis_obj(blis_num_threads, BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE);
        bli_setsc(gemm_beta, 0.0, &blis_obj.beta);
        bli_rntm_set_ways(1, 1, blis_num_threads, 1, 1, &blis_obj.rntm);
#endif

        unsigned int image_merge_count = images/thread_qty;
        if (image_merge_count_rem && (omp_get_thread_num() < image_merge_count_rem)) {
            image_merge_count++;
        }

        int threadOffset = (omp_get_thread_num() * image_merge_count);
        if (image_merge_count_rem) {
            threadOffset = (omp_get_thread_num() * (images/thread_qty + 1));
            if (omp_get_thread_num() > image_merge_count_rem) {
                threadOffset = (omp_get_thread_num() * (images/thread_qty) +
                                image_merge_count_rem);
            }
        }
        unsigned long inputOffset = ((unsigned long)channels*height*width*threadOffset);
        unsigned long outputOffset = ((unsigned long)ldc*
                                      (out_height*out_width)* threadOffset);

        unsigned int merge_loop = image_merge_count;
        unsigned long gemmRows = ((unsigned long)width_col* height_col * merge_loop);

        unsigned int offset = filter_offset;

        //printf("M=%ld\tN=%ld\tK=%ld\n", gemmRows, no_of_filter, channels*kernel_h*kernel_w);
#if BLIS_EXPERT
        bli_obj_create_with_attached_buffer(blis_obj.dt, gemmRows,
                                            channels*kernel_h*kernel_w,
                                            data_col+inputOffset, channels*kernel_h*kernel_w, 1, &blis_obj.a);
        bli_obj_create_with_attached_buffer(blis_obj.dt, channels*kernel_h*kernel_w,
                                            no_of_filter,
                                            (void *)filter, no_of_filter, 1, &blis_obj.b);
        bli_obj_create_with_attached_buffer(blis_obj.dt, gemmRows, no_of_filter,
                                            out_layer+outputOffset+offset, ldc, 1, &blis_obj.c);

        bli_gemm_ex(&blis_obj.alpha, &blis_obj.a, &blis_obj.b, &blis_obj.beta,
                    &blis_obj.c, NULL, &blis_obj.rntm);
#else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, gemmRows, no_of_filter,
                    channels*kernel_h*kernel_w, 1.0f,
                    data_col+inputOffset, channels*kernel_h*kernel_w, filter, no_of_filter,
                    gemm_beta,
                    out_layer+outputOffset+offset, ldc);
#endif

        //Below Bias and activation code can be eliminated if not required
        unsigned long biasOffset = outputOffset+offset;
        zenPostOps(zenEnvObj, out_layer, elementwise_input,gemmRows, 1, no_of_filter,
                   ldc, biasOffset,
                   bias, relu, false, scale,
                   blis_num_threads);
    }
#if 0
    gettimeofday(&end, 0);
    float elapsed;
    elapsed = timedifference_msec(start, end);
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2D_bestSmallGemmVer2, no_of_images=",
               no_of_images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_h=", pad_h, " pad_w=", pad_w,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " Time=", elapsed, "ms");
#endif

}


//This implementation is based on im2col and gemm(BLIS) where im2col is performed on input
//images/featureMap one by one followed by gemm call to blis which computes the feature map for the i/p image
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//This ALGO performs best when input height and width > 20
//For height and width < 20, spiltting adds overhead for GEMM calls(causes more GEMM calls on samll sizes)
void zenConvolution2DsmallGemmSplit(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int images,
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
    const bool sum_fused,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2DsmallGemmSplit, no_of_images=",
               images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " isConcat=", concat, " filter_offset=", filter_offset,
               " total_filters=", total_filters);

    float gemm_beta = 0.0;
    if (sum_fused) {
        gemm_beta = 1.0;
    }

    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    int height_col = out_height;
    int width_col = out_width;

    //Merging image tranformation height in order to make SGEMM M close to N
    //int merge_height1 = ((no_of_filter/height_col)==0?1:(no_of_filter/height_col));

    //performance varies with Differnt merge height
    //Merging image tranformation height in order to make SGEMM's M close to
    //BLIS_SMALL_MATRIX, BLIS_SMALL_MATRIX size chunk get optimial perf with
    //BLIS on ROME, currently tested with 24, 32, 48 and 64 no. of cores
    //TODO: Need to check the same with MILAN, and test with other core combination
    //TODO: need to test various values of merge height with all models
    int merge_height1 = height_col;

    //For MILAN BLIS_SMALL_MATRIX_MILAN is the optimal factor for split
    // Currently tested for INT8 path and good boost with 1st layer
    //TODO: Test with fp32 path for perf improvement with CNN models, expected
    //  performance gains(10-40%) for fist later but no luck,
    //  However no regression with fp32 patch on ROME.
    if (zendnn_getenv_int("ZENDNN_INT8_SUPPORT") == 1) {
        merge_height1 = (BLIS_SMALL_MATRIX_MILAN/height_col)?
                        (BLIS_SMALL_MATRIX_MILAN/height_col):1;
    }
    else {
        merge_height1 = (BLIS_SMALL_MATRIX/height_col)?
                        (BLIS_SMALL_MATRIX/height_col):1;
    }

    unsigned long data_col_size = (((unsigned long)kernel_h*kernel_w*channels *
                                    width_col *
                                    merge_height1)*sizeof(float)*thread_qty);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);

    float *data_col = NULL;
    int zenLibPoolEnable = zenEnvObj.zenLibMemPoolEnable;
    ZenLibMemoryPool *zenLibPoolBuffer;

    //ZenLibMemPool Optimization reuse tmp buffers from the pool. By default
    //  its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
    //  pool optimization
    //  Cases where buffers in pool are not free or requested size is more
    //  than available buffer size in Pool, control will fall back to
    //  default way of allocation
    if (zenLibPoolEnable) {
        zenLibPoolBuffer = ZenLibMemoryPool::getZenLibMemPool(0);
        if (zenLibPoolBuffer) {
            int status = zenLibPoolBuffer->acquireZenLibPoolBuf(&data_col, data_col_size,
                         1);
            if (status) {
                zenLibPoolEnable = false;
            }
        }
        else {
            zenLibPoolEnable = false;
        }
    }
    if (!zenLibPoolEnable) {
        data_col = (float *)aligned_alloc(ALIGNED_OFFSET, data_col_size);
    }
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DsmallGemmSplit Memory Error while allocating patch matrix");
        return;
    }
    //Need to optimize transpose if used for NON HWCN filter format
    float *filterNew = (float *)
                       filter;//transpose(filter, no_of_filter, kernel_h*kernel_w*channels);

    int blis_num_threads = 1;
#if BLIS_EXPERT
    //Enable Nested parallelism when patch matrix outer loop is not able to use all threads
    //TODO: Need to try with Forced nested parallelism for all sizes, try with various blis_num_threads values
    if (thread_qty > images) {
        blis_num_threads = (thread_qty%images)==0?(thread_qty/images):((
                               thread_qty/images)+1);
    }
    else {
        blis_num_threads = thread_qty<1?thread_qty:1;
    }
    thread_qty = (thread_qty%blis_num_threads)==0?(thread_qty/blis_num_threads):
                 (thread_qty/blis_num_threads)+1;
    omp_set_max_active_levels(2);
#else
    if (thread_qty > images) {
        thread_qty = images;
    }
    omp_set_max_active_levels(1);
#endif

    unsigned int ldc = no_of_filter;
    if (concat) {
        ldc = total_filters;
    }

    #pragma omp parallel num_threads(thread_qty)
    {
#if BLIS_EXPERT
        if ((thread_qty%blis_num_threads)!=0 && omp_get_num_threads()==(thread_qty-1)) {
            blis_num_threads = thread_qty%blis_num_threads;
        }
        //creating blis expert interface
        blis_expert blis_obj(blis_num_threads, BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE);
        //bli_rntm_set_ways( blis_num_threads, 1, 1, 1, 1, &blis_obj.rntm );
        bli_setsc(gemm_beta, 0.0, &blis_obj.beta);
#endif
        unsigned int loopCount = (images%thread_qty)==0 ? images/thread_qty :
                                 (images/thread_qty)+1;
        for (int i=0; i<loopCount; i++) {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            if (threadOffset >= images) {
                break;
            }
            //Patch matrix formation splitted along height_col
            //We merge height rows based on merge factor
            int merge_height =  merge_height1;

            unsigned long inputOffset = ((unsigned long)channels*height*width*threadOffset);
            unsigned long patchInputOffset = (((unsigned long)kernel_h*kernel_w*channels*
                                               width_col*merge_height) * omp_get_thread_num());

            if ((kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
                    out_width == width)) {
                patchInputOffset = inputOffset;
            }
            unsigned long outputOffset = ((unsigned long)ldc*
                                          (out_height*out_width)* threadOffset);

            float *data_col_tmp = data_col + patchInputOffset;
            unsigned int data_col_offset = 0;
            int merge_count = 0;

            int h = 0;
            int h_pad = -pad_t;
            //Im2row tranformation
            for (h = 0; h < height_col; ++h) {
                //if (!(kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
                //      out_width == width)) {

                int w_pad = -pad_l;
                if (merge_count == 0) {
                    data_col_offset = 0;
                }

                //To enable unrolling and better vectorization, (depth == 3) path unrolled the loop
                //along channel, this address first layer of convolution where no. of channels
                //in Image is 3.
                //For (depth%8 == 0), common case for other conv layers, vectorization is enabled by
                //to tell compiler to generate AVX256 SIMD instruction using (simd_blocks*8) loop.
                //Observed perf improvement with googlenet and alexnet.

                if (channels == 3) {
                    for (int w = 0; w < width_col; ++w) {
#if 1
                        //Unrolling of inner loop for kernel_w * kernel_h
                        int ih = h_pad;
                        int iw = w_pad;
                        bool unrool_flag = (ih < (h_pad + kernel_h)) && (ih >= 0) &&
                                           ((ih + kernel_h) < height) && (iw < (w_pad + kernel_w)) && (iw >= 0) &&
                                           ((iw + kernel_w) < width);
                        if ((kernel_h == 3)  && unrool_flag) {

                            unsigned long offset = (inputOffset) + (ih * width + iw) * channels;
                            im2row_unrool_3x3(data_col_tmp, data_col_offset, in_layer, offset);
                            ++ih;

                            data_col_offset += 9;
                            offset = (inputOffset) + (ih * width + iw) * channels;
                            im2row_unrool_3x3(data_col_tmp, data_col_offset, in_layer, offset);
                            ++ih;

                            data_col_offset += 9;
                            offset = (inputOffset) + (ih * width + iw) * channels;
                            im2row_unrool_3x3(data_col_tmp, data_col_offset, in_layer, offset);
                            ++ih;
                            data_col_offset += 9;

                        }
                        else if ((kernel_h == 7)  && unrool_flag) {

                            unsigned long offset = (inputOffset) + (ih * width + iw) * channels;
                            im2row_unrool_7x3(data_col_tmp, data_col_offset, in_layer, offset);
                            ++ih;

                            data_col_offset += 21;
                            offset = (inputOffset) + (ih * width + iw) * channels;
                            im2row_unrool_7x3(data_col_tmp, data_col_offset, in_layer, offset);
                            ++ih;

                            data_col_offset += 21;
                            offset = (inputOffset) + (ih * width + iw) * channels;
                            im2row_unrool_7x3(data_col_tmp, data_col_offset, in_layer, offset);
                            ++ih;

                            data_col_offset += 21;
                            offset = (inputOffset) + (ih * width + iw) * channels;
                            im2row_unrool_7x3(data_col_tmp, data_col_offset, in_layer, offset);
                            ++ih;

                            data_col_offset += 21;
                            offset = (inputOffset) + (ih * width + iw) * channels;
                            im2row_unrool_7x3(data_col_tmp, data_col_offset, in_layer, offset);
                            ++ih;

                            data_col_offset += 21;
                            offset = (inputOffset) + (ih * width + iw) * channels;
                            im2row_unrool_7x3(data_col_tmp, data_col_offset, in_layer, offset);
                            ++ih;

                            data_col_offset += 21;
                            offset = (inputOffset) + (ih * width + iw) * channels;
                            im2row_unrool_7x3(data_col_tmp, data_col_offset, in_layer, offset);
                            ++ih;

                            data_col_offset += 21;
                        }
                        else {
                            for (int ih = h_pad; ih < h_pad + kernel_h; ++ih) {
                                for (int iw = w_pad; iw < w_pad + kernel_w; ++iw) {
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                        unsigned long offset = (inputOffset) + (ih * width + iw) * channels;
                                        //#pragma omp simd
                                        //for (int k=0; k<channels; k++) {
                                        data_col_tmp[data_col_offset + 0] = in_layer[offset + 0];
                                        data_col_tmp[data_col_offset + 1] = in_layer[offset + 1];
                                        data_col_tmp[data_col_offset + 2] = in_layer[offset + 2];
                                        //}
                                    }
                                    else {
                                        // This should be simply padded with zero.
                                        //#pragma omp simd
                                        //for (int k=0; k<channels; k++) {
                                        data_col_tmp[data_col_offset + 0] = 0;
                                        data_col_tmp[data_col_offset + 1] = 0;
                                        data_col_tmp[data_col_offset + 2] = 0;
                                        //}
                                    }
                                    data_col_offset += channels;
                                }
                            }
                        }
#else
                        for (int ih = h_pad; ih < h_pad + kernel_h; ++ih) {
                            for (int iw = w_pad; iw < w_pad + kernel_w; ++iw) {
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    unsigned long offset = (inputOffset) + (ih * width + iw) * channels;
                                    //#pragma omp simd
                                    //for (int k=0; k<channels; k++) {
                                    data_col_tmp[data_col_offset + 0] = in_layer[offset + 0];
                                    data_col_tmp[data_col_offset + 1] = in_layer[offset + 1];
                                    data_col_tmp[data_col_offset + 2] = in_layer[offset + 2];
                                    //}
                                }
                                else {
                                    // This should be simply padded with zero.
                                    //#pragma omp simd
                                    //for (int k=0; k<channels; k++) {
                                    data_col_tmp[data_col_offset + 0] = 0;
                                    data_col_tmp[data_col_offset + 1] = 0;
                                    data_col_tmp[data_col_offset + 2] = 0;
                                    //}
                                }
                                data_col_offset += channels;
                            }
                        }
#endif
                        w_pad += stride_w;
                    }
                }
                else if ((channels%8) == 0) {
                    int simd_blocks = channels/8;
                    for (int w = 0; w < width_col; ++w) {
                        for (int ih = h_pad; ih < h_pad + kernel_h; ++ih) {
                            for (int iw = w_pad; iw < w_pad + kernel_w; ++iw) {
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    unsigned long offset = (inputOffset) + (ih * width + iw) * channels;
                                    #pragma omp simd
                                    for (int k=0; k<simd_blocks*8; k++) {
                                        data_col_tmp[data_col_offset + k] = in_layer[offset + k];
                                    }
                                }
                                else {
                                    // This should be simply padded with zero.
                                    #pragma omp simd
                                    for (int k=0; k<simd_blocks*8; k++) {
                                        data_col_tmp[data_col_offset + k] = 0;
                                    }
                                }
                                data_col_offset += channels;
                            }
                        }
                        w_pad += stride_w;
                    }

                }
                else {
                    for (int w = 0; w < width_col; ++w) {
                        for (int ih = h_pad; ih < h_pad + kernel_h; ++ih) {
                            for (int iw = w_pad; iw < w_pad + kernel_w; ++iw) {
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    unsigned long offset = (inputOffset) + (ih * width + iw) * channels;
                                    #pragma omp simd
                                    for (int k=0; k<channels; k++) {
                                        data_col_tmp[data_col_offset + k] = in_layer[offset + k];
                                    }
                                }
                                else {
                                    // This should be simply padded with zero.
                                    #pragma omp simd
                                    for (int k=0; k<channels; k++) {
                                        data_col_tmp[data_col_offset + k] = 0;
                                    }
                                }
                                data_col_offset += channels;
                            }
                        }
                        w_pad += stride_w;
                    }
                }
                //}
                //else
                //  data_col_tmp = data_col + patchInputOffset +
                //(channels*width_col*(h-(merge_height-1)));

                merge_count++;
                if (h==(height_col -1) && !((height_col%merge_height)==0)) {
                    merge_height = height_col%merge_height;
                }
                if (merge_count == merge_height) {

                    unsigned long offset = ((unsigned long)ldc*width_col*(h-
                                            (merge_height-1))) + filter_offset;
#if BLIS_EXPERT
                    bli_obj_create_with_attached_buffer(blis_obj.dt, width_col*merge_height,
                                                        channels*kernel_h*kernel_w,
                                                        data_col_tmp, channels*kernel_h*kernel_w, 1, &blis_obj.a);
                    bli_obj_create_with_attached_buffer(blis_obj.dt, channels*kernel_h*kernel_w,
                                                        no_of_filter,
                                                        (void *)filter, no_of_filter, 1, &blis_obj.b);
                    bli_obj_create_with_attached_buffer(blis_obj.dt, width_col*merge_height,
                                                        no_of_filter,
                                                        out_layer+outputOffset + offset, ldc, 1, &blis_obj.c);

                    bli_gemm_ex(&blis_obj.alpha, &blis_obj.a, &blis_obj.b, &blis_obj.beta,
                                &blis_obj.c, NULL, &blis_obj.rntm);
#else
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, width_col*merge_height,
                                no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                                data_col_tmp, channels*kernel_h*kernel_w, filterNew, no_of_filter, gemm_beta,
                                out_layer + outputOffset + offset, ldc);
#endif

                    unsigned biasOffset = outputOffset + offset;
                    zenPostOps(zenEnvObj, out_layer, elementwise_input,width_col, merge_height,
                               no_of_filter, ldc,
                               biasOffset, bias,
                               relu, false, scale, blis_num_threads);
                    merge_count = 0;
                }
                h_pad += stride_h;
            }
        }
    }
//If ZenMemPool Optimization is enabled(default), update the state of
//  Memory pool based on input_array address
    if (zenLibPoolEnable) {
        zenLibPoolBuffer->zenLibMemPoolFree((float *)data_col);
    }
    else {
        free(data_col);
    }
}



//This implementation is based on im2col and gemm(BLIS) where im2col is performed on input
//images/featureMap one by one followed by gemm call to blis which computes the feature map for the i/p image
//and then add bias value on it.
//I/p and o/p format will be NCHW
//Multi thread parallization happen at OMP level for im2col and bias operation
//For gemm, BLIS will take care of the parallelism
//We call it threading inside BLIS
//TODO: Add inplace concat support
void zenConvolution2DlatencyVer1(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int   images,
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
    const int out_width,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters

) {

    unsigned int thread_qty = zenEnvObj.omp_num_threads;

    //#New Implementation with im2col and gemm function.
    //im2col().....parallel version is also available

    unsigned long data_col_size = ((kernel_h*kernel_w*channels)*
                                   (out_height*out_width)*sizeof(float)*images);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = (float *)aligned_alloc(ALIGNED_OFFSET, data_col_size);

    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DlatencyVer1 Memory Error while allocating patch matrix");
        return;
    }

    //Divide filters into channel groups based on the number of threads
    int channel_group =  thread_qty;
    int remainder = no_of_filter % (channel_group);
    int out_ch_per_group = (no_of_filter-remainder)/(channel_group);

    for (int i=0; i<images; i++) {
        unsigned long bufferOffset = ((unsigned long)(kernel_h*kernel_w*channels)*
                                      (out_height*out_width) * i);
        im2col_parNCHW(in_layer+(channels*height*width*i), channels, height, width,
                       kernel_h,
                       kernel_w, pad_h, pad_w, stride_h, stride_w, data_col+bufferOffset);
        int w_offset = kernel_h * kernel_w * channels;
        int o_h_w    = out_height*out_width;
        //Run channel groupwise parallel convolution
        #pragma omp parallel for
        for (int itr=0; itr <= channel_group; itr++) {
            if (itr < channel_group) {
                int weight_offset = itr * out_ch_per_group *  w_offset;
                int out_offset = (i*channels + itr *out_ch_per_group)* o_h_w;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            out_ch_per_group, o_h_w, w_offset, 1.0F, filter + weight_offset,
                            w_offset, data_col + bufferOffset, o_h_w, 0.0F, out_layer + out_offset, o_h_w);
            }
            else {
                int weight_offset = out_ch_per_group * (channel_group) *  w_offset;
                int out_offset = (i*no_of_filter + itr * out_ch_per_group)* o_h_w;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            remainder, o_h_w, w_offset, 1.0F, filter + weight_offset,
                            w_offset, data_col + bufferOffset, o_h_w, 0.0F, out_layer + out_offset, o_h_w);
            }
        }
        unsigned int outOffset = (no_of_filter*(out_height*out_width)*i);
        if (bias != NULL) {
            #pragma omp parallel for num_threads(thread_qty)
            for (int r=0; r<no_of_filter; r++) {
                for (int m=0; m<out_height*out_width; m++) {
                    out_layer[outOffset + (r*(out_height*out_width)) + m]+= bias[r];
                }
            }
        }
    }
    free(data_col);

#if 0
    gettimeofday(&end, 0);
    float elapsed;
    elapsed = timedifference_msec(start, end);
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2D_LatencyVer1, no_of_images=",
               no_of_images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_h=", pad_h, " pad_w=", pad_w, " stride_h=", stride_h,
               " stride_w=",stride_w, " Time=", elapsed, "ms");
#endif

}


//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner for im2row and bias operation
//For gemm, BLIS will take care of the parallelism
//We call it mix of threading inside and outside BLIS
//TODO: Add inplace concat support
void zenConvolution2DlatencyVer2(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int images,
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
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters

) {

#if 0
    unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) {
        thread_qty = 1;
    }

    struct timeval start, end;
    gettimeofday(&start, 0);
#endif

    unsigned int thread_qty = zenEnvObj.omp_num_threads;


    unsigned int loopCount =
        images;//(no_of_images%thread_qty)==0 ? no_of_images/thread_qty : (no_of_images/thread_qty)+1;

    unsigned long data_col_size = ((unsigned long)(kernel_h*kernel_w*channels)*
                                   (out_height*out_width)*sizeof(float)*images);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col ;
    if (!(kernel_h ==1 && kernel_w==1 && out_height ==height &&
            out_width == width)) {
        data_col = (float *)aligned_alloc(ALIGNED_OFFSET, data_col_size);
    }
    else {
        data_col = (float *)in_layer;
    }


    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DlatencyVer2 Memory Error while allocating patch matrix");
        return;
    }

    //#pragma omp parallel num_threads(thread_qty)
    {
        for (int i=0; i<loopCount; i++) {
            int threadOffset = i;//omp_get_thread_num()+ (i*thread_qty);
            //if(threadOffset >= no_of_images)
            //  break;
            unsigned long bufferOffset = (unsigned long)(kernel_h*kernel_w*channels)*
                                         (out_height*out_width) * threadOffset;
            unsigned long inputOffset = (unsigned long)channels*height*width*threadOffset;
            //unsigned int bufferOffset = ((kernel_h*kernel_w*channels)*(out_height*out_width) * omp_get_thread_num());

            //im2row is more efficient than im2col with NHWC
            if (!(kernel_h ==1 && kernel_w==1 && out_height ==height &&
                    out_width == width)) {
                im2rowNHWC_par(in_layer+inputOffset, channels, height, width, kernel_h,
                               kernel_w,
                               pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, data_col + bufferOffset);
            }
            unsigned long outBufferOffset = ((unsigned long)no_of_filter*
                                             (out_height*out_width)*
                                             threadOffset);
            //AMD BLIS bases matrix multiplication
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out_height*out_width,
                        no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                        data_col+bufferOffset, channels*kernel_h*kernel_w, filter, no_of_filter, 0.0f,
                        out_layer+outBufferOffset, no_of_filter);


            if (bias && !relu) {
                #pragma omp parallel for num_threads(thread_qty)
                for (int m=0; m<out_height*out_width; m++)
                    for (int r=0; r<no_of_filter; r++) {
                        out_layer[i*out_height*out_width*no_of_filter + (m*(no_of_filter)) + r]+=
                            bias[r];
                    }
            }

            if (bias && relu) {
                #pragma omp parallel for num_threads(thread_qty)
                for (int m=0; m<out_height*out_width; m++)
                    for (int r=0; r<no_of_filter; r++) {
                        out_layer[i*out_height*out_width*no_of_filter + (m*(no_of_filter)) +
                                  r]+=bias[r];
                        if (out_layer[i*out_height*out_width*no_of_filter + (m*(no_of_filter)) + r] <
                                0) {
                            out_layer[i*out_height*out_width*no_of_filter + (m*(no_of_filter)) + r] = 0;
                        }
                    }
            }
        }
    }
    if (!(kernel_h ==1 && kernel_w==1 && out_height ==height &&
            out_width == width)) {
        free(data_col);
    }

#if 0
    gettimeofday(&end, 0);
    float elapsed;
    elapsed = timedifference_msec(start, end);
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2D_LatencyVer2, no_of_images=",
               no_of_images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_h=", pad_h, " pad_w=", pad_w,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " Time=", elapsed, "ms");
#endif

}


//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner for im2row and bias operation
//For gemm, BLIS will take care of the parallelism
//We call it mix of threading inside and outside BLIS
//TODO: Add inplace concat support
void zenConvolution2DlatencyVer3(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int images,
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
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters

) {

#if 0
    unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) {
        thread_qty = 1;
    }

    struct timeval start, end;
    gettimeofday(&start, 0);
#endif
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2DlatencyVer3, no_of_images=",
               images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w);
    unsigned int thread_qty = zenEnvObj.omp_num_threads;

    int height_col =
        out_height;//(height + pad_h + pad_w - kernel_h) / stride_h + 1;
    int width_col = out_width;//(width + pad_h + pad_w - kernel_w) / stride_w + 1;

    int blis_num_threads = 1;
#if BLIS_EXPERT
    //Enable Nested parallelism when patch matrix outer loop is not able to use all threads
    //TODO: Need to try with Forced nested parallelism for all sizes, try with various blis_num_threads values
    if (height_col < thread_qty) {
        blis_num_threads = thread_qty/height_col;
    }
    thread_qty = (thread_qty%blis_num_threads)==0?(thread_qty/blis_num_threads):
                 (thread_qty/blis_num_threads)+1;
    omp_set_max_active_levels(2);
#else
    omp_set_max_active_levels(1);
#endif

    int threads = height_col<thread_qty?height_col:thread_qty;
    unsigned long data_col_size = ((unsigned long)(kernel_h*kernel_w*channels)*
                                   (out_width)*sizeof(float)*threads);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = (float *)aligned_alloc(ALIGNED_OFFSET, data_col_size);
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DlatencyVer3 Memory Error while allocating patch matrix");
        return;
    }

    #pragma omp parallel for num_threads(threads)
    for (int k=0; k<height_col; k++) {

        //If inner_threads*threads < OMP_NUM_THREADS, inner_threads will be incremented for few parent threads
        //This make sure that all the threads are utilized
        //TODO: Check trade off with this for deeper layers where input size is small
        int temp = zenEnvObj.omp_num_threads - (threads*blis_num_threads);
        int inner_threads = blis_num_threads;
        if (omp_get_thread_num() < temp) {
            inner_threads++;
        }

#if BLIS_EXPERT
        blis_expert blis_obj(inner_threads, BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE);
#endif
        //im2row is more efficient than im2col with NHWC
        unsigned long patchHeightOffset = (unsigned long)omp_get_thread_num()
                                          *width_col*(kernel_h*kernel_w*channels);

        im2rowNHWCsplit(in_layer, channels, height, width, kernel_h, kernel_w, pad_t,
                        pad_l, pad_b, pad_r,
                        stride_h, stride_w, data_col + patchHeightOffset, 1, k, inner_threads);

#if BLIS_EXPERT
        bli_obj_create_with_attached_buffer(blis_obj.dt, width_col,
                                            channels*kernel_h*kernel_w,
                                            data_col+patchHeightOffset, channels*kernel_h*kernel_w, 1, &blis_obj.a);
        bli_obj_create_with_attached_buffer(blis_obj.dt, channels*kernel_h*kernel_w,
                                            no_of_filter,
                                            (void *)filter, no_of_filter, 1, &blis_obj.b);
        bli_obj_create_with_attached_buffer(blis_obj.dt, width_col, no_of_filter,
                                            out_layer+(width_col*no_of_filter*k), no_of_filter, 1, &blis_obj.c);

        bli_gemm_ex(&blis_obj.alpha, &blis_obj.a, &blis_obj.b, &blis_obj.beta,
                    &blis_obj.c, NULL, &blis_obj.rntm);
#else
        //AMD BLIS bases matrix multiplication
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, width_col, no_of_filter,
                    channels*kernel_h*kernel_w, 1.0f,
                    data_col+patchHeightOffset, channels*kernel_h*kernel_w, filter, no_of_filter,
                    0.0f,
                    out_layer+(width_col*no_of_filter*k), no_of_filter);
#endif

        unsigned long biasOffset = (width_col*no_of_filter*k);
        zenPostOps(zenEnvObj, out_layer, elementwise_input,width_col, 1, no_of_filter,
                   no_of_filter, biasOffset,
                   bias, relu, false, scale,
                   inner_threads,0,0,images);
    }
    free(data_col);

#if 0
    gettimeofday(&end, 0);
    float elapsed;
    elapsed = timedifference_msec(start, end);
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2D_LatencyVer3, no_of_images=",
               no_of_images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_h=", pad_h, " pad_w=", pad_w,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " Time=", elapsed, "ms");
#endif

}

//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner for im2row and bias operation
//For gemm, BLIS will take care of the parallelism
//We call it mix of threading inside and outside BLIS
void zenConvolution2DlatencyVer4(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int images,
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
    const bool sum_fused,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

#if 0
    unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) {
        thread_qty = 1;
    }

    struct timeval start, end;
    gettimeofday(&start, 0);
#endif

    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2DlatencyVer4, no_of_images=",
               images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " isConcat=", concat, " filter_offset=", filter_offset,
               " total_filters=", total_filters);

    float gemm_beta = 0.0;
    if (sum_fused) {
        gemm_beta = 1.0;
    }

    unsigned int thread_qty = zenEnvObj.omp_num_threads;

    int height_col =
        out_height;//(height + pad_h + pad_w - kernel_h) / stride_h + 1;
    int width_col = out_width;//(width + pad_h + pad_w - kernel_w) / stride_w + 1;

    int blis_num_threads = 1;
#if BLIS_EXPERT
    //Enable Nested parallelism when patch matrix outer loop is not able to use all threads
    //TODO: Need to try with Forced nested parallelism for all sizes, try with various blis_num_threads values
    if (height_col < thread_qty) {
        blis_num_threads = thread_qty/height_col;
    }
    thread_qty = (thread_qty%blis_num_threads)==0?(thread_qty/blis_num_threads):
                 (thread_qty/blis_num_threads)+1;
    omp_set_max_active_levels(2);
#else
    omp_set_max_active_levels(1);
#endif

    int threads = height_col<thread_qty?height_col:thread_qty;
    unsigned int loopCount = height_col/threads;
    unsigned int height_rem = height_col%threads;
    unsigned int height_alloc_count = (height_col%threads==0)?1:2;

    unsigned long data_col_size = ((unsigned long)(kernel_h*kernel_w*channels)*
                                   (out_width)*height_alloc_count*sizeof(float)*threads);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = NULL;
    int zenLibPoolEnable = zenEnvObj.zenLibMemPoolEnable;
    ZenLibMemoryPool *zenLibPoolBuffer;

    if ((kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
            out_width == width)) {
        data_col = (float *)in_layer;
    }
    else {

        //ZenLibMemPool Optimization reuse tmp buffers from the pool. By default
        //  its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
        //  pool optimization
        //  Cases where buffers in pool are not free or requested size is more
        //  than available buffer size in Pool, control will fall back to
        //  default way of allocation
        if (zenLibPoolEnable) {
            zenLibPoolBuffer = ZenLibMemoryPool::getZenLibMemPool(0);
            if (zenLibPoolBuffer) {
                int status = zenLibPoolBuffer->acquireZenLibPoolBuf(&data_col, data_col_size,
                             1);
                if (status) {
                    zenLibPoolEnable = false;
                }
            }
            else {
                zenLibPoolEnable = false;
            }
        }
        if (!zenLibPoolEnable) {
            data_col = (float *)aligned_alloc(ALIGNED_OFFSET, data_col_size);
        }

    }
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DlatencyVer4 Memory Error while allocating patch matrix");
        return;
    }
    unsigned int ldc = no_of_filter;
    if (concat) {
        ldc = total_filters;
    }

    #pragma omp parallel num_threads(threads)
    {
        int inner_threads = blis_num_threads;
#if BLIS_EXPERT
        //If inner_threads*threads < OMP_NUM_THREADS, inner_threads will be incremented for few parent threads
        //This make sure that all the threads are utilized
        //TODO: Check trade off with this for deeper layers where input size is small
        int temp = zenEnvObj.omp_num_threads - (threads*blis_num_threads);
        if (omp_get_thread_num() < temp) {
            inner_threads++;
        }
        blis_expert blis_obj(inner_threads, BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE);
        bli_setsc(gemm_beta, 0.0, &blis_obj.beta);
#endif
        //unsigned int loopCount = (height_col%threads)==0 ? height_col/threads : (height_col/threads)+1;
        int height_count = 1;
        for (int i=0; i<loopCount; i++) {
            int threadOffset = omp_get_thread_num()+ (i*threads);
            if (height_rem && (i==(loopCount-1))) {
                if (omp_get_thread_num()<height_rem) {
                    height_count++;
                    threadOffset = (omp_get_thread_num()*height_count)+ (i*threads);
                }
                else {
                    threadOffset += height_rem;
                }

            }
            if (threadOffset >= height_col) {
                break;
            }
            unsigned long patchHeightOffset = (unsigned long)threadOffset
                                              *width_col*(kernel_h*kernel_w*channels);

            if (!(kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
                    out_width == width)) {
                patchHeightOffset = (unsigned long)omp_get_thread_num()
                                    *width_col*height_alloc_count*(kernel_h*kernel_w*channels);
                im2rowNHWCsplit(in_layer, channels, height, width, kernel_h, kernel_w, pad_t,
                                pad_l, pad_b, pad_r,
                                stride_h, stride_w, data_col + patchHeightOffset, height_count, threadOffset,
                                inner_threads);
            }
            unsigned long outputOffset = ((unsigned long)width_col*ldc*threadOffset) +
                                         filter_offset;
#if BLIS_EXPERT
            bli_obj_create_with_attached_buffer(blis_obj.dt, width_col*height_count,
                                                channels*kernel_h*kernel_w,
                                                data_col+patchHeightOffset, channels*kernel_h*kernel_w, 1, &blis_obj.a);
            bli_obj_create_with_attached_buffer(blis_obj.dt, channels*kernel_h*kernel_w,
                                                no_of_filter,
                                                (void *)filter, no_of_filter, 1, &blis_obj.b);
            bli_obj_create_with_attached_buffer(blis_obj.dt, width_col*height_count,
                                                no_of_filter,
                                                out_layer+outputOffset, ldc, 1, &blis_obj.c);

            bli_gemm_ex(&blis_obj.alpha, &blis_obj.a, &blis_obj.b, &blis_obj.beta,
                        &blis_obj.c, NULL, &blis_obj.rntm);
#else
            //AMD BLIS bases matrix multiplication
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, width_col*height_count,
                        no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                        data_col+patchHeightOffset, channels*kernel_h*kernel_w, filter, no_of_filter,
                        gemm_beta,
                        out_layer+outputOffset, ldc);
#endif

            unsigned long biasOffset = outputOffset;
            zenPostOps(zenEnvObj, out_layer, elementwise_input,width_col, height_count,
                       no_of_filter, ldc,
                       biasOffset, bias,
                       relu, false, scale, inner_threads);
        }
    }
    if (!(kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
            out_width == width)) {
        //If ZenMemPool Optimization is enabled(default), update the state of
        //  Memory pool based on input_array address
        if (zenLibPoolEnable) {
            zenLibPoolBuffer->zenLibMemPoolFree((float *)data_col);
        }
        else {
            free(data_col);
        }
    }

#if 0
    gettimeofday(&end, 0);
    float elapsed;
    elapsed = timedifference_msec(start, end);
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2D_LatencyVer3, no_of_images=",
               no_of_images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_h=", pad_h, " pad_w=", pad_w,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " Time=", elapsed, "ms");
#endif

}


//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//For gemm, BLIS will take care of the parallelism
//We call it mix of threading inside and outside BLIS
void zenConvolution2DlatencyVer5(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int images,
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
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

#if 0
    unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) {
        thread_qty = 1;
    }

    struct timeval start, end;
    gettimeofday(&start, 0);
#endif
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2DlatencyVer5, no_of_images=",
               images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " isConcat=", concat, " filter_offset=", filter_offset,
               " total_filters=", total_filters);

    unsigned int thread_qty = zenEnvObj.omp_num_threads;

    int height_col =
        out_height;//(height + pad_h + pad_w - kernel_h) / stride_h + 1;
    int width_col = out_width;//(width + pad_h + pad_w - kernel_w) / stride_w + 1;

    int blis_num_threads = 1;
#if BLIS_EXPERT
    //Enable Nested parallelism when patch matrix outer loop is not able to use all threads
    //TODO: Need to try with Forced nested parallelism for all sizes, try with various blis_num_threads values
    if (height_col < thread_qty) {
        blis_num_threads = thread_qty/height_col;
    }
    thread_qty = (thread_qty%blis_num_threads)==0?(thread_qty/blis_num_threads):
                 (thread_qty/blis_num_threads)+1;
    omp_set_max_active_levels(2);
#else
    omp_set_max_active_levels(1);
#endif

    int threads = height_col<thread_qty?height_col:thread_qty;
    unsigned int height_merge_count = (height_col%threads==0)?(height_col/threads):
                                      (height_col/threads +1);
    int height_col_rem = height_col%threads;

    unsigned long data_col_size = ((unsigned long)(kernel_h*kernel_w*channels)*
                                   (out_width)*sizeof(float)*thread_qty*height_merge_count);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = (float *)aligned_alloc(ALIGNED_OFFSET, data_col_size);
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DlatencyVer5 Memory Error while allocating patch matrix");
        return;
    }

    unsigned int ldc = no_of_filter;
    if (concat) {
        ldc = total_filters;
    }

    #pragma omp parallel num_threads(threads) private(height_merge_count)
    {
        //This force to compute convolution in single iteration with all threads
        //Merges outer loop of patch matrix
        height_merge_count = height_col/threads;
        if (height_col_rem && omp_get_thread_num() < height_col_rem) {
            height_merge_count++;
        }

        int threadOffset = omp_get_thread_num() * height_merge_count;
        if (height_col_rem) {
            threadOffset = (omp_get_thread_num() * (height_col/threads + 1));
            if (omp_get_thread_num() >= height_col_rem) {
                threadOffset = (omp_get_thread_num() * (height_col/threads) + height_col_rem);
            }
        }
        unsigned long patchHeightOffset = (unsigned long)threadOffset*width_col*
                                          (kernel_h*kernel_w*channels);
        unsigned long outputOffset = ((unsigned long)ldc * out_width *
                                      threadOffset) + filter_offset;

        im2rowNHWCsplit(in_layer, channels, height, width, kernel_h, kernel_w, pad_t,
                        pad_l, pad_b, pad_r,
                        stride_h, stride_w, data_col + patchHeightOffset, height_merge_count,
                        threadOffset, blis_num_threads);

#if BLIS_EXPERT
        blis_expert blis_obj(blis_num_threads, BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE);
        bli_obj_create_with_attached_buffer(blis_obj.dt, width_col*height_merge_count,
                                            channels*kernel_h*kernel_w,
                                            data_col+patchHeightOffset, channels*kernel_h*kernel_w, 1, &blis_obj.a);
        bli_obj_create_with_attached_buffer(blis_obj.dt, channels*kernel_h*kernel_w,
                                            no_of_filter,
                                            (void *)filter, no_of_filter, 1, &blis_obj.b);
        bli_obj_create_with_attached_buffer(blis_obj.dt, width_col*height_merge_count,
                                            no_of_filter,
                                            out_layer+outputOffset, ldc, 1, &blis_obj.c);

        bli_gemm_ex(&blis_obj.alpha, &blis_obj.a, &blis_obj.b, &blis_obj.beta,
                    &blis_obj.c, NULL, &blis_obj.rntm);
#else
        //AMD BLIS bases matrix multiplication
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    width_col*height_merge_count, no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                    data_col+patchHeightOffset, channels*kernel_h*kernel_w, filter, no_of_filter,
                    0.0f,
                    out_layer+outputOffset, ldc);
#endif

        unsigned long biasOffset = outputOffset;
        zenPostOps(zenEnvObj, out_layer,elementwise_input, width_col*height_merge_count,
                   1,
                   no_of_filter, ldc, biasOffset,
                   bias, relu, false, scale, blis_num_threads);
    }

    free(data_col);
#if 0
    gettimeofday(&end, 0);
    float elapsed;
    elapsed = timedifference_msec(start, end);
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2D_LatencyVer3, no_of_images=",
               no_of_images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_h=", pad_h, " pad_w=", pad_w,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " Time=", elapsed, "ms");
#endif

}


//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner
//This Algo handles cases where input height and eidth > 300
//Splitting favours better cache utilization during patch matrix formation followed by GEMM calls
void zenConvolution2DsmallGemmSplitLatency(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int images,
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
    const bool sum_fused,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

    zendnnInfo(ZENDNN_ALGOLOG,
               "zenConvolution2DsmallGemmSplitLatency, no_of_images=", images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " isConcat=", concat, " filter_offset=", filter_offset,
               " total_filters=", total_filters);

    float gemm_beta = 0.0;
    if (sum_fused) {
        gemm_beta = 1.0;
    }

    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    int height_col = out_height;
    int width_col = out_width;

    //splitFactor decides the width_col splitChunkSize
    int splitFactor = 2;
    int splitChunkSize = (width_col%splitFactor==0)?(width_col/splitFactor):
                         (width_col/splitFactor)+1;

    //Need to optimize transpose if used for NON HWCN filter format
    float *filterNew = (float *)
                       filter;//transpose(filter, no_of_filter, kernel_h*kernel_w*channels);

    int blis_num_threads = 1;
#if BLIS_EXPERT
    //Enable Nested parallelism when patch matrix outer loop is not able to use all threads
    //TODO: Need to try with Forced nested parallelism for all sizes, try with various blis_num_threads values
    if (height_col < thread_qty) {
        blis_num_threads = thread_qty/height_col;
    }
    thread_qty = (thread_qty%blis_num_threads)==0?(thread_qty/blis_num_threads):
                 (thread_qty/blis_num_threads)+1;
    omp_set_max_active_levels(2);
#else
    omp_set_max_active_levels(1);
#endif

    int col_width = ((width + pad_l + pad_r - kernel_w) / stride_w + 1)*kernel_h*
                    channels * kernel_w;
    int h = 0;
    int h_pad = -pad_t;

    int threads = height_col<thread_qty?height_col:thread_qty;
    unsigned long data_col_size = ((unsigned long)(
                                       kernel_h*kernel_w*channels*splitChunkSize)
                                   *sizeof(float)*threads);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = (float *)aligned_alloc(ALIGNED_OFFSET, data_col_size);
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DsmallGemmSplitLatency Memory Error while allocating patch matrix");
        return;
    }
    float *col_data_old = data_col;

    unsigned int ldc = no_of_filter;
    if (concat) {
        ldc = total_filters;
    }

    #pragma omp parallel for num_threads(threads) private(h, data_col, h_pad)
    for (h = 0; h < height_col; ++h) {

        int inner_threads = blis_num_threads;
#if BLIS_EXPERT
        //If inner_threads * threads < OMP_NUM_THREADS, inner_threads will be incremented for few parent threads
        //This make sure that all the threads are utilized
        //TODO: Check trade off with this for deeper layers where input size is small
        int temp = zenEnvObj.omp_num_threads - (threads*blis_num_threads);
        if (omp_get_thread_num() < temp) {
            inner_threads++;
        }
        blis_expert blis_obj(inner_threads, BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE);
        bli_setsc(gemm_beta, 0.0, &blis_obj.beta);
#endif

        int out_count = 0;
        int chunkNo = 0;
        int w_pad = -pad_l;
        h_pad = -pad_t + (h * stride_h);

        unsigned long patchHeightOffset = (unsigned long)omp_get_thread_num()
                                          *splitChunkSize*(kernel_h*kernel_w*channels);
        data_col = col_data_old + patchHeightOffset;;

        //Im2row tranformation
        for (int w = 0; w < width_col; ++w) {
            for (int ih = h_pad; ih < h_pad + kernel_h; ++ih) {
                for (int iw = w_pad; iw < w_pad + kernel_w; ++iw) {
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        int offset = (ih * width + iw) * channels;
                        #pragma omp simd
                        for (int k=0; k<channels; k++) {
                            data_col[k] = in_layer[offset + k];
                        }
                    }
                    else {
                        // This should be simply padded with zero.
                        #pragma omp simd
                        for (int k=0; k<channels; k++) {
                            data_col[k] = 0;
                        }
                    }
                    data_col += channels;
                }
            }
            w_pad += stride_w;
            out_count++;
            if ((w+1)%splitChunkSize == 0 || (w == width_col-1)) {
                int outOffset = (ldc*(width_col*h)+
                                 (ldc*splitChunkSize*chunkNo) + filter_offset);
#if BLIS_EXPERT
                bli_obj_create_with_attached_buffer(blis_obj.dt, out_count,
                                                    channels*kernel_h*kernel_w,
                                                    col_data_old + patchHeightOffset, channels*kernel_h*kernel_w, 1, &blis_obj.a);
                bli_obj_create_with_attached_buffer(blis_obj.dt, channels*kernel_h*kernel_w,
                                                    no_of_filter, (void *)filterNew, no_of_filter, 1, &blis_obj.b);
                bli_obj_create_with_attached_buffer(blis_obj.dt, out_count, no_of_filter,
                                                    out_layer+outOffset, ldc, 1, &blis_obj.c);

                bli_gemm_ex(&blis_obj.alpha, &blis_obj.a, &blis_obj.b, &blis_obj.beta,
                            &blis_obj.c, NULL, &blis_obj.rntm);
#else
                //AMD BLIS based matrix multiplication
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out_count, no_of_filter,
                            channels*kernel_h*kernel_w, 1.0f,
                            col_data_old + patchHeightOffset, channels*kernel_h*kernel_w, filterNew,
                            no_of_filter, gemm_beta,
                            out_layer + outOffset, ldc);
#endif
                unsigned biasOffset = outOffset;
                zenPostOps(zenEnvObj, out_layer, elementwise_input,out_count, 1, no_of_filter,
                           ldc, biasOffset,
                           bias, relu, false, scale,
                           inner_threads);

                data_col = col_data_old + patchHeightOffset;
                out_count = 0;
                chunkNo++;
            }
        }
        //h_pad += stride_h;
    }
    data_col = col_data_old;
    free(data_col);
}



//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format is NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner
//This Algo handles SMALL GEMM cases where patch matrix exposes M(rows of first matrix) of SGEMM < no_of_filter
//Merging reduces the no. of GEMM calls by merging multiple images
void zenConvolution2DsmallGemmMergeLatency(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int images,
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
    const bool sum_fused,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

    zendnnInfo(ZENDNN_ALGOLOG,
               "zenConvolution2DsmallGemmMergeLatency, no_of_images=", images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=", stride_w,
               " isConcat=", concat, " filter_offset=", filter_offset,
               " total_filters=", total_filters);

    float gemm_beta = 0.0;
    if (sum_fused) {
        gemm_beta = 1.0;
    }

    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    unsigned int height_col = out_height;
    unsigned int width_col = out_width;

    //mergeFactor decides the height_col mergeChunkSize
    unsigned int mergeFactor =  1;
    unsigned int mergeFactor1 =
        2048/channels;   //2048 is choosen because BLIS sgemm call get sweet spot based on this

    mergeFactor = mergeFactor1 > height_col ? height_col : mergeFactor1;
    unsigned int no_of_merge_chunk = (height_col%mergeFactor)==0?
                                     (height_col/mergeFactor):((height_col/mergeFactor)+1);

    //Need to optimize transpose if used for NON HWCN filter format
    float *filterNew = (float *)
                       filter;//transpose(filter, no_of_filter, kernel_h*kernel_w*channels);

    //If total thread is not able to consume outer layer of patch matrix, nester parallelism will be enabled
    //Setting blis threads for nested parallelism
    int blis_num_threads = (thread_qty/no_of_merge_chunk) <= 0 ? 1 :
                           (thread_qty/no_of_merge_chunk);

    int h = 0;
    int h_pad = -pad_t;

    unsigned int threads = no_of_merge_chunk < thread_qty?no_of_merge_chunk:
                           thread_qty;
    unsigned long data_col_size = (((unsigned long)
                                    kernel_h*kernel_w*channels*mergeFactor*width_col)*sizeof(float)*thread_qty);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);

    float *data_col = NULL;
    int zenLibPoolEnable = zenEnvObj.zenLibMemPoolEnable;
    ZenLibMemoryPool *zenLibPoolBuffer;

    //ZenLibMemPool Optimization reuse tmp buffers from the pool. By default
    //  its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
    //  pool optimization
    //  Cases where buffers in pool are not free or requested size is more
    //  than available buffer size in Pool, control will fall back to
    //  default way of allocation
    if (zenLibPoolEnable) {
        zenLibPoolBuffer = ZenLibMemoryPool::getZenLibMemPool(0);
        if (zenLibPoolBuffer) {
            int status = zenLibPoolBuffer->acquireZenLibPoolBuf(&data_col, data_col_size,
                         1);
            if (status) {
                zenLibPoolEnable = false;
            }
        }
        else {
            zenLibPoolEnable = false;
        }
    }
    if (!zenLibPoolEnable) {
        data_col = (float *)aligned_alloc(ALIGNED_OFFSET, data_col_size);
    }


    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DsmallGemmMergeLatency Memory Error while allocating patch matrix");
        return;
    }
    float *col_data_old = data_col;
#if BLIS_EXPERT
    omp_set_max_active_levels(2);
#else
    omp_set_max_active_levels(1);
#endif

    unsigned int ldc = no_of_filter;
    if (concat) {
        ldc = total_filters;
    }

    #pragma omp parallel for num_threads(threads) private(h, data_col, h_pad)
    for (int i = 0; i< no_of_merge_chunk; ++i) {

        unsigned int inner_threads = blis_num_threads;
#if BLIS_EXPERT
        //If inner_threads * threads < OMP_NUM_THREADS, inner_threads will be incremented for few parent threads
        //This make sure that all the threads are utilized
        //TODO: Check trade off with this for deeper layers where input size is small
        int temp = zenEnvObj.omp_num_threads - (threads*blis_num_threads);
        if (omp_get_thread_num() < temp) {
            inner_threads++;
        }
        blis_expert blis_obj(inner_threads, BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE);
        bli_setsc(gemm_beta, 0.0, &blis_obj.beta);
#endif
        unsigned int mergeChunkSize = mergeFactor;
        if (i==(no_of_merge_chunk-1) && (height_col%mergeFactor != 0)) {
            mergeChunkSize = height_col%mergeFactor;
        }
        unsigned long patchHeightOffset = (unsigned long)omp_get_thread_num()
                                          *mergeFactor*width_col*(kernel_h*kernel_w*channels);
        data_col = col_data_old + patchHeightOffset;
        //Im2row tranformation
        for (int j = 0; j < mergeChunkSize; ++j) {
            h = i*mergeFactor + j;
            int w_pad = -pad_l;
            h_pad = -pad_t + (h * stride_h);
            for (int w = 0; w < width_col; ++w) {
                for (int ih = h_pad; ih < h_pad + kernel_h; ++ih) {
                    for (int iw = w_pad; iw < w_pad + kernel_w; ++iw) {
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            int offset = (ih * width + iw) * channels;
                            #pragma omp simd
                            for (int k=0; k<channels; k++) {
                                data_col[k] = in_layer[offset + k];
                            }
                        }
                        else {
                            // This should be simply padded with zero.
                            #pragma omp simd
                            for (int k=0; k<channels; k++) {
                                data_col[k] = 0;
                            }
                        }
                        data_col += channels;
                    }
                }
                w_pad += stride_w;
            }
            if (j == (mergeChunkSize-1)) {
                unsigned int outOffset = (ldc*width_col*(h-(mergeChunkSize-1)) + filter_offset);
#if BLIS_EXPERT
                bli_obj_create_with_attached_buffer(blis_obj.dt, width_col*mergeChunkSize,
                                                    channels*kernel_h*kernel_w,
                                                    col_data_old+patchHeightOffset, channels*kernel_h*kernel_w, 1, &blis_obj.a);
                bli_obj_create_with_attached_buffer(blis_obj.dt, channels*kernel_h*kernel_w,
                                                    no_of_filter,
                                                    (void *)filter, no_of_filter, 1, &blis_obj.b);
                bli_obj_create_with_attached_buffer(blis_obj.dt, width_col*mergeChunkSize,
                                                    no_of_filter,
                                                    out_layer+outOffset, ldc, 1, &blis_obj.c);

                bli_gemm_ex(&blis_obj.alpha, &blis_obj.a, &blis_obj.b, &blis_obj.beta,
                            &blis_obj.c, NULL, &blis_obj.rntm);
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, width_col*mergeChunkSize,
                            no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                            col_data_old + patchHeightOffset, channels*kernel_h*kernel_w, filterNew,
                            no_of_filter, gemm_beta,
                            out_layer + outOffset, ldc);
#endif

                unsigned biasOffset = outOffset;
                zenPostOps(zenEnvObj, out_layer, elementwise_input,width_col, mergeChunkSize,
                           no_of_filter, ldc,
                           biasOffset, bias,
                           relu, false, scale, inner_threads,0,0,images);
                data_col = col_data_old + patchHeightOffset;
            }
        }
    }
    data_col = col_data_old;

    //If ZenMemPool Optimization is enabled(default), update the state of
    //  Memory pool based on input_array address
    if (zenLibPoolEnable) {
        zenLibPoolBuffer->zenLibMemPoolFree((float *)data_col);
    }
    else {
        free(data_col);
    }
}


//An umbrella C++ interface for zendnn convolution
void zenConvolution2Dgemm(
    const float *in_layer,
    const int batchsize,
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
    const bool sum_fused,
    const float *scale,
    const float *elementwise_input,
    const bool concat = false,
    const int filter_offset = 0,
    const int total_filters = 0
) {

    //TODO: This should be part of zendnn initialization
    zendnnEnv zenEnvObj = readEnv();

    struct timeval start, end;
    gettimeofday(&start, 0);

    if (batchsize > 1) {
        //Throughput path BS > 1
#if DIRECT_CONV_GEMV
        //This is direct convolution which is GEMV and sdot base...currently not optimized
        //For input height > 14 and < 224 this may perform better
        //if (!(kernel_h == 1 && kernel_w == 1 &&  out_height == height && out_width == width)) {
        if ((kernel_h == 3 && kernel_w == 3 &&  height > 14 && height < 224)) {
            zenConvolution2D_direct(zenEnvObj, in_layer, batchsize, channels, height, width,
                                    filter, no_of_filter,
                                    kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                                    out_layer, out_height, out_width, relu, scale);
        }
        else {
            if ((kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
                    out_width == width)) {
                zenConvolution2DsmallGemm1x1(zenEnvObj, in_layer, batchsize, channels, height,
                                             width, filter, no_of_filter,
                                             kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, i
                                             out_layer, out_height, out_width, relu, scale);
            }
            else if (out_height*out_width < no_of_filter) {
                //if out_height*out_width < no_of_filter, M will be less than N
                //for sgemm execution, for thease cases we merge two images for BS>1
                //and increase the size of M, this gives optimal performance with sgemm
                zenConvolution2DsmallGemmMerge(zenEnvObj, in_layer, batchsize, channels, height,
                                               width, filter, no_of_filter,
                                               kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                                               out_layer, out_height, out_width, relu, scale);
            }
            else {
                //zenConvolution2DsmallGemmVer2(zenEnvObj, in_layer, batchsize, channels, height, width, filter, no_of_filter,
                zenConvolution2D_directVer3(zenEnvObj, in_layer, batchsize, channels, height,
                                            width, filter, no_of_filter,
                                            kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                                            out_layer, out_height, out_width, relu, scale);
            }
        }
#else

#if WINOGRAD_CONV
        //TODO: extend winograd to uneven padding as well
        //TODO: Need to get more data form diffent model to tune CONV_BIG_SIZE and CONV_INPUT_HEIGHT better
        //TODO: Need to check the same for non uniform height x width
        //CONV_INPUT_SIZE and CONV_INPUT_HEIGHT is based on the heuristics of googlenet resnet and vgg
        //TODO: Tune CONV_INPUT_SIZE CONV_INPUT_HEIGHT for other models too
        //TODO: Need to support winograd version for ZenInceptionOp. Currenlty if we force winograd
        //version for googlenet variants the accuracy validation will fail.
        if (stride_h == 1 && stride_w == 1 && kernel_h == 3 && kernel_w == 3 &&
                height % 2 == 0 && width % 2 == 0 && (concat == false)
                && (height*channels >= CONV_INPUT_SIZE) && (height<CONV_INPUT_HEIGHT)) {
            winograd_2x2_3x3(zenEnvObj, in_layer, batchsize, channels, height, width,
                             filter, no_of_filter, kernel_h, kernel_w,
                             pad_t, pad_l, pad_b, pad_r,
                             bias,
                             out_layer, out_height, out_width,
                             relu, sum_fused, scale);
        }
        else
#endif
            if ((kernel_h != 1 && kernel_w != 1 && out_height*out_width >= no_of_filter)) {
                //This ALGO performs best when input height and width > 20
                //For height and width < 20, spiltting adds overhead for GEMM calls(causes more GEMM calls on samll sizes)
                zenConvolution2DsmallGemmSplit(zenEnvObj, in_layer, batchsize, channels, height,
                                               width, filter, no_of_filter,
                                               kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                                               out_layer, out_height, out_width, relu, sum_fused, scale, elementwise_input,
                                               concat, filter_offset, total_filters);
            }
#if 0
        //This Algo handles 1x1 kernel where patch matrix formation is not required
            else if ((kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
                      out_width == width)) {
                //TF MemPool optimization complementing this kermnel rather going for seperate
                //zenConvolution2DsmallGemm1x1 kernel
                if (height > SPLIT_CONV_INPUT)
                    zenConvolution2DsmallGemmVer2(zenEnvObj, in_layer, batchsize, channels, height,
                                                  width, filter, no_of_filter,
                                                  kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                                                  out_layer, out_height, out_width, relu, sum_fused, scale, elementwise_input,
                                                  concat, filter_offset, total_filters);
                else
                    //This Algo handles 1x1 kernel where patch matrix formation is not required
                    //TODO This path should work best for all cases. BLIS team is working on it
                    //zenConvolution2DGemm1x1Direct(zenEnvObj, in_layer, batchsize, channels, height,
                    zenConvolution2DsmallGemm1x1(zenEnvObj, in_layer, batchsize, channels, height,
                                                 width, filter, no_of_filter,
                                                 kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                                                 out_layer, out_height, out_width, relu, sum_fused, scale, elementwise_input,
                                                 concat, filter_offset, total_filters);


            }
        //With ROME, where L3 cache(16M) is shared by 4 cores, zenConvolution2DsmallGemmVer2 works best
        //TODO: check with other m/c configuration and its impact
            else if (out_height*out_width < no_of_filter && kernel_h <=3 && kernel_w <= 3) {
                //This Algo handles SMALL GEMM cases where patch matrix exposes M of SGEMM < no_of_filter
                //Merging reduces the no. of GEMM calls by merging multiple images
                zenConvolution2DsmallGemmMerge(zenEnvObj, in_layer, batchsize, channels, height,
                                               width, filter, no_of_filter,
                                               kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                                               out_layer, out_height, out_width, relu, sum_fused, scale, elementwise_input,
                                               concat, filter_offset, total_filters);
                //TODO: Need to try zenConvolution2Dbase for this branch with all models
                //zenConvolution2Dbase(zenEnvObj, in_layer, batchsize, channels, height, width, filter, no_of_filter,
            }
#endif
            else {
                zenConvolution2DsmallGemmVer2(zenEnvObj, in_layer, batchsize, channels, height,
                                              width, filter, no_of_filter,
                                              kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                                              out_layer,
                                              out_height, out_width, relu, sum_fused, scale, elementwise_input,
                                              concat, filter_offset, total_filters);
                //TODO: Need to try zenConvolution2Dbase for this branch with all models
                //zenConvolution2Dbase(zenEnvObj, in_layer, batchsize, channels, height, width, filter, no_of_filter,
            }
#endif
    }
    else {
        //Latency path BS == 1
        if ((kernel_h == 1 && kernel_w == 1 &&  out_height == height &&
                out_width == width))
            //This Algo handles 1x1 kernel where patch matrix formation is not required
            if (0)//height > SPLIT_CONV_INPUT)//for some sizes this patch is better
                //TODO Need to find the right switch between below paths
                zenConvolution2DlatencyVer4(zenEnvObj, in_layer, batchsize, channels, height,
                                            width, filter, no_of_filter,
                                            kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                                            out_layer, out_height, out_width, relu, sum_fused, scale, elementwise_input,
                                            concat, filter_offset, total_filters);
            else
                zenConvolution2DGemm1x1Direct(zenEnvObj, in_layer, batchsize, channels, height,
                                              width, filter, no_of_filter,
                                              kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                                              out_layer, out_height, out_width, relu, sum_fused, scale, elementwise_input,
                                              concat, filter_offset, total_filters);

        else if (0)
            //TODO: Need to try zenConvolution2DsmallGemmSplitLatency if i/p height or width > 300
            zenConvolution2DsmallGemmSplitLatency(zenEnvObj, in_layer, batchsize, channels,
                                                  height, width, filter, no_of_filter,
                                                  kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                                                  out_layer, out_height, out_width, relu, sum_fused, scale, elementwise_input,
                                                  concat, filter_offset, total_filters);
        else if (height < SMALL_CONV_INPUT && kernel_h == 3 && kernel_w == 3)
            //Merging reduces the no. of GEMM calls by merging multiple inner loop during patch matrix formation
            //This works well with filter size 3
            //TODO Tyy this with other filter sizes with different models
            zenConvolution2DsmallGemmMergeLatency(zenEnvObj, in_layer, batchsize, channels,
                                                  height, width, filter, no_of_filter,
                                                  kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                                                  out_layer, out_height, out_width, relu, sum_fused, scale, elementwise_input,
                                                  concat, filter_offset, total_filters);
        //TODO: Try this version for this path with all models....may work optimial with specific i/p sizes
        //zenConvolution2DlatencyVer5(..)
        else
            zenConvolution2DlatencyVer4(zenEnvObj, in_layer, batchsize, channels, height,
                                        width, filter, no_of_filter,
                                        kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                                        out_layer,
                                        out_height, out_width, relu, sum_fused, scale, elementwise_input,
                                        concat, filter_offset, total_filters);


    }

    gettimeofday(&end, 0);
    float elapsed;
    elapsed = timedifference_msec(start, end);
    zendnnInfo(ZENDNN_PROFLOG, "zenConvolution2D_gemm, no_of_images=", batchsize,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " isConcat=", concat, " filter_offset=", filter_offset,
               " total_filters=", total_filters,
               " Time=", elapsed, "ms");
}

void zenConvolution2D(
    const float *in_layer,
    const int batchsize,
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
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

    //TODO: perform other checks...eg. for all input dimansions
    if ((in_layer == NULL)|| (filter == NULL) || (out_layer == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2D Memory is not defined for in_layer or filter or out_layer");
        return;
    }
    //int pad_t,pad_l,pad_b,pad_r;
    //compute_padding(pad_h,pad_w,&pad_t,&pad_l,&pad_b,&pad_r);
    zenConvolution2Dgemm(in_layer, batchsize, channels, height, width, filter,
                         no_of_filter,
                         kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, NULL,
                         out_layer, out_height, out_width, 0/*relu*/, false/*sum_fused*/, NULL /*scale*/,
                         NULL/*elementwise*/,
                         concat, filter_offset, total_filters);


}

void zenConvolution2DwithBias(
    const float *in_layer,
    const int batchsize,
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
    const bool concat,
    const int filter_offset,
    const int total_filters
) {
    //TODO: perform other checks...eg. for all input dimansions
    if ((in_layer == NULL)|| (filter == NULL) || (out_layer == NULL) ||
            (bias == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DwithBias Memory is not defined for in_layer or filter or out_layer");
        return;
    }
    //int pad_t,pad_l,pad_b,pad_r;
    //compute_padding(pad_h,pad_w,&pad_t,&pad_l,&pad_b,&pad_r);
    zenConvolution2Dgemm(in_layer, batchsize, channels, height, width, filter,
                         no_of_filter,
                         kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                         out_layer, out_height, out_width, 0/*relu*/, false/*sum_fused*/, NULL /*scale*/,
                         NULL/*elementwise*/,
                         concat, filter_offset, total_filters);
}

void zenConvolution2DwithBiasRelu(
    const float *in_layer,
    const int batchsize,
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
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

    //TODO: perform other checks...eg. for all input dimansions
    if ((in_layer == NULL)|| (filter == NULL) || (out_layer == NULL) ||
            (bias == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DwithBiasRelu Memory is not defined for in_layer or filter or out_layer");
        return;
    }
    //int pad_t,pad_l,pad_b,pad_r;
    //compute_padding(pad_h,pad_w,&pad_t,&pad_l,&pad_b,&pad_r);
    zenConvolution2Dgemm(in_layer, batchsize, channels, height, width, filter,
                         no_of_filter,
                         kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                         out_layer, out_height, out_width, 1, false/*sum_fused*/, NULL, NULL,
                         concat, filter_offset, total_filters);
}

void zenConvolution2DwithBiasSum(
    const float *in_layer,
    const int batchsize,
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
    const bool concat,
    const int filter_offset,
    const int total_filters
) {
    //TODO: perform other checks...eg. for all input dimansions
    if ((in_layer == NULL)|| (filter == NULL) || (out_layer == NULL) ||
            (bias == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DwithBiasSum Memory is not defined for in_layer or filter or out_layer");
        return;
    }
    //int pad_t,pad_l,pad_b,pad_r;
    //compute_padding(pad_h,pad_w,&pad_t,&pad_l,&pad_b,&pad_r);
    zenConvolution2Dgemm(in_layer, batchsize, channels, height, width, filter,
                         no_of_filter,
                         kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                         out_layer, out_height, out_width, 0/*relu*/, true/*sum_fused*/, NULL /*scale*/,
                         NULL/*elementwise*/,
                         concat, filter_offset, total_filters);
}

void zenConvolution2DwithBiasSumRelu(
    const float *in_layer,
    const int batchsize,
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
    const bool concat,
    const int filter_offset,
    const int total_filters
) {

    //TODO: perform other checks...eg. for all input dimansions
    if ((in_layer == NULL)|| (filter == NULL) || (out_layer == NULL) ||
            (bias == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DwithBiasSumRelu Memory is not defined for in_layer or filter or out_layer");
        return;
    }
    //int pad_t,pad_l,pad_b,pad_r;
    //compute_padding(pad_h,pad_w,&pad_t,&pad_l,&pad_b,&pad_r);
    zenConvolution2Dgemm(in_layer, batchsize, channels, height, width, filter,
                         no_of_filter,
                         kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                         out_layer, out_height, out_width, 1, true/*sum_fused*/, NULL, NULL,
                         concat, filter_offset, total_filters);
}


void zenConvolution2DwithBatchNorm(
    const float *in_layer,
    const int batchsize,
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
    const bool concat,
    const int filter_offset,
    const int total_filters

) {

    //TODO: perform other checks...eg. for all input dimansions
    if ((in_layer == NULL)|| (filter == NULL) || (out_layer == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DwithBatchNorm Memory is not defined for in_layer or filter or out_layer");
        return;
    }
    //int pad_t,pad_l,pad_b,pad_r;
    //compute_padding(pad_h,pad_w,&pad_t,&pad_l,&pad_b,&pad_r);
    float *bias = (float *)malloc(sizeof(float)*no_of_filter);
    #pragma omp parallel for
    for (int r=0; r <no_of_filter; r++) {
        bias[r] = offset[r]-(scale[r]*mean[r]);
    }

    zenConvolution2Dgemm(in_layer, batchsize, channels, height, width, filter,
                         no_of_filter,
                         kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                         out_layer, out_height, out_width, 0/*relu*/, false/*sum_fused*/, scale,
                         NULL/*elementwise*/,
                         concat, filter_offset, total_filters);
    //zenBatchNorm(batchsize, out_height, out_width,no_of_filter,scale,mean,offset,out_layer, 1,0);
    free(bias);
}

void zenConvolution2DwithBatchNormRelu(
    const float *in_layer,
    const int batchsize,
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
    const bool concat,
    const int filter_offset,
    const int total_filters

) {

    //TODO: perform other checks...eg. for all input dimansions
    if ((in_layer == NULL)|| (filter == NULL) || (out_layer == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DwithBatchNormRelu Memory is not defined for in_layer or filter or out_layer");
        return;
    }
    //int pad_t,pad_l,pad_b,pad_r;
    //compute_padding(pad_h,pad_w,&pad_t,&pad_l,&pad_b,&pad_r);
    float *bias = (float *)malloc(sizeof(float)*no_of_filter);
    #pragma omp parallel for
    for (int r=0; r <no_of_filter; r++) {
        bias[r] = offset[r]-(scale[r]*mean[r]);
    }
    zenConvolution2Dgemm(in_layer, batchsize, channels, height, width, filter,
                         no_of_filter,
                         kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                         out_layer, out_height, out_width, 1, false/*sum_fused*/, scale,
                         NULL/*elementwise*/,
                         concat, filter_offset, total_filters);
    //zenBatchNorm(batchsize, out_height, out_width,no_of_filter,scale,mean,offset,out_layer, 1,1);
    free(bias);
}

void zenConvolution2DwithBatchNormsum(
    const float *in_layer,
    const int batchsize,
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
    const float *elementwise_input,
    float *out_layer,
    const int out_height,
    const int out_width,
    const bool concat,
    const int filter_offset,
    const int total_filters

) {

    //Need to perforam other checks...eg. for all input dimansions
    if ((in_layer == NULL)|| (filter == NULL) || (out_layer == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DwithBatchNormRelu Memory is not defined for in_layer or filter or out_layer");
        return;
    }
    //int pad_t,pad_l,pad_b,pad_r;
    //compute_padding(pad_h,pad_w,&pad_t,&pad_l,&pad_b,&pad_r);
    float *bias = (float *)malloc(sizeof(float)*no_of_filter);

    //parallel bias calculation
    #pragma omp parallel for
    for (int r=0; r <no_of_filter; r++) {
        bias[r] = offset[r]-(scale[r]*mean[r]);
    }
    zenConvolution2Dgemm(in_layer, batchsize, channels, height, width, filter,
                         no_of_filter,
                         kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                         out_layer, out_height, out_width, 1, false/*sum_fused*/, scale,
                         elementwise_input,
                         concat, filter_offset, total_filters);
    free(bias);
}


