/*******************************************************************************
* Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <omp.h>
#include <cblas.h>
#include <time.h>
#include "zendnn_logging.hpp"
#include "zendnn_helper.hpp"

using namespace zendnn;

#define ALIGNED_OFFSET          64
//#define BLIS_SMALL_MATRIX     680
#define BLIS_SMALL_MATRIX       784 //Based on network layer i/p
#define BLIS_SMALL_MATRIX2      196 //Based on network layer i/p

#define BLIS_SMALL_MATRIX_COUNT     2
#define BLIS_SMALL_MATRIX2_COUNT    6
#define BLIS_SMALL_MATRIX3_COUNT    12


//This implementation is based on im2col and gemm(BLIS) where im2col is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NCHW
//Multi thread parallization happen at OMP level for im2col and bias and gemm operation
//We call it threading outside BLIS
void zenConvolution2D_ver2(
    const float *in_layer,  // float or char input?
    const int no_of_images,
    const int channels,
    const int height,
    const int width,
    const float *filter,
    const int no_of_filter,
    //const int channels,           //no. of channels is same as no. of channels filters
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const float *bias,
    float *out_layer,       //o/p to this function
    //unsigned char *out_layer,     // float or char?
    //const int out_no_of_images,
    //const int out_channels,       // same as no. of filters
    const int out_height,           //o/p to this function
    const int out_width             //o/p to this function
) {
    zendnnInfo(ZENDNN_ALGOLOG, "zenConvolution2D ver2 [zendnn convolution]");
#if 1
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) thread_qty = 1;

    //Need to change this for latency optimization
    if (thread_qty > no_of_images) {
        thread_qty = no_of_images;
    }


    unsigned int loopCount = (no_of_images%thread_qty)==0 ?
                             no_of_images/thread_qty : (no_of_images/thread_qty)+1;
    //if(no_of_images > cpuVitualCores)
    //  bufferBucket = cpuVitualCores;

    unsigned long size = (kernel_h*kernel_w*channels)*(out_height*out_width) *
                         thread_qty;
    float *data_col = (float *)malloc(size * sizeof(float));
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2D_ver2 Memory Error while allocating patch matrix");
        return;
    }


    #pragma omp parallel num_threads(thread_qty)
    {
        for (int i=0; i<loopCount; i++) {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            if (threadOffset >= no_of_images) {
                break;
            }
            //unsigned int bufferOffset = ((kernel_h*kernel_w*channels)*(out_height*out_width) * threadOffset);
            unsigned long bufferOffset = ((kernel_h*kernel_w*channels)*
                                          (out_height*out_width) * omp_get_thread_num());

            unsigned long outBufferOffset = (no_of_filter* (out_height*out_width)*
                                             threadOffset);

            //im2col_par(in_layer+(channels*height*width*i), channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, data_col + ((kernel_h*kernel_w*channels)*(out_height*out_width)) * threadOffset);
            //im2col(in_layer+(channels*height*width*threadOffset), channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, data_col + ((kernel_h*kernel_w*channels)*(out_height*out_width)) * threadOffset);
            im2col_parNCHW(in_layer+(channels*height*width*threadOffset), channels, height,
                           width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                           data_col + bufferOffset);
            //im2colNCHW(in_layer+(channels*height*width*threadOffset), channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, data_col + bufferOffset);

            //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, no_of_filter, out_height*out_width, channels*kernel_h*kernel_w, 1.0f,
            //                      filter, channels*kernel_h*kernel_w, data_col, out_height*out_width, 0.0f, out_layer+(no_of_filter*(out_height*out_width)*omp_get_thread_num()), out_height*out_width);

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, no_of_filter,
                        out_height*out_width, channels*kernel_h*kernel_w, 1.0f,
                        filter, channels*kernel_h*kernel_w, data_col + bufferOffset,
                        out_height*out_width, 0.0f, out_layer + outBufferOffset, out_height*out_width);



            if (bias != NULL) {
                for (int r=0; r<no_of_filter; r++) {
                    for (int m=0; m<out_height*out_width; m++) {
                        out_layer[outBufferOffset + (r*(out_height*out_width)) + m]+= bias[r];
                    }
                }
            }

        }
    }
    free(data_col);

#else
#endif
}



//This implementation is based on kn2row and gemm(BLIS) where im2col is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//http://www.prime-project.org/wp-content/uploads/sites/206/2018/02/Talk-10-David-Gregg-Parallel-Multi-Channel-Convolution-using-General-Matrix-Multiplication.pdf
//I/p and o/p format will be NCHW
//Multi thread parallization happen at OMP level for im2col and bias and gemm operation
//We call it threading outside BLIS
void convolution2D_ver3(
    //const unsigned char* in_layer,
    const float *in_layer,  // float or char input?
    const int no_of_images,
    const int channels,
    const int height,
    const int width,
    const float *filter,
    const int no_of_filter,
    //const int channels,           //no. of channels is same as no. of channels filters
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const float *bias,
    float *out_layer,       //o/p to this function
    //unsigned char *out_layer,     // float or char?
    //const int out_no_of_images,
    //const int out_channels,       // same as no. of filters
    const int out_height,           //o/p to this function
    const int out_width             //o/p to this function
) {

#if 1
    //unsigned short cpuVitualCores = get_nprocs();
    unsigned short bufferBucket = no_of_images;
    //if (no_of_images > cpuVitualCores) {
    //    bufferBucket = cpuVitualCores;
    //}

    //#New Implementation with im2col and gemm function.
    //im2col().....parallel version is also available
    //unsigned int  size = (kernel_h*kernel_w*channels)*(out_height*out_width) * no_of_images;
    unsigned long  filterMatrixSize = (kernel_h*kernel_w*no_of_filter) * channels;
    float *filter_row = (float *)malloc(filterMatrixSize * sizeof(float));

    unsigned long  outMatrixSize = (kernel_h*kernel_w*no_of_filter)* (height*width);
    float *outMatrix = (float *)malloc(outMatrixSize * sizeof(float));
    float *outMatrixNew = (float *)malloc(no_of_filter * height*width * sizeof(
            float));
    if (filter_row == NULL || outMatrix == NULL || outMatrixNew == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "convolution2D_ver3 Memory Error while allocating patch matrix");
        return;
    }

    //NCHW2HWNC(filter, no_of_filter, channels, kernel_h, kernel_w, filter_row);

    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) thread_qty = 1;
    unsigned int loopCount = (no_of_images%thread_qty)==0 ?
                             no_of_images/thread_qty : (no_of_images/thread_qty)+1;
    #pragma omp parallel num_threads(thread_qty)
    {
        for (int i=0; i<loopCount; i++) {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            if (threadOffset >= no_of_images) {
                break;
            }
            //unsigned int bufferOffset = ((kernel_h*kernel_w*channels)*(out_height*out_width) * threadOffset);
            unsigned  long bufferOffset = ((height*width*channels) * threadOffset);
            unsigned long outBufferOffset = (no_of_filter* (out_height*out_width)*
                                             threadOffset);


            //im2col_par(in_layer+(channels*height*width*i), channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, data_col + ((kernel_h*kernel_w*channels)*(out_height*out_width)) * threadOffset);
            //im2col(in_layer+(channels*height*width*threadOffset), channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, data_col + bufferOffset);

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        kernel_h*kernel_w*no_of_filter, height*width, channels, 1.0f,
                        filter_row, channels, in_layer + bufferOffset, height*width, 0.0f, outMatrix,
                        height*width);
            //filter_row, kernel_h*kernel_w*no_of_filter, in_layer + bufferOffset, channels, 0.0f, outMatrix, kernel_h*kernel_w*no_of_filter);



            //shiftAdd(outMatrix, kernel_h*kernel_w, no_of_filter, height, width, outMatrixNew, out_layer+outBufferOffset, out_height, out_width, kernel_h, kernel_w );

        }
    }
    free(filter_row);
    free(outMatrix);

#else
#endif
}



//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner for im2row and bias operation
//For gemm, BLIS will take care of the parallelism
//We call it mix of threading inside and outside BLIS
void zenConvolution2D_ver4(
    //const unsigned char* in_layer,
    const float *in_layer,  // float or char input?
    const int no_of_images,
    const int channels,
    const int height,
    const int width,
    const float *filter,
    const int no_of_filter,
    //const int channels,           //no. of channels is same as no. of channels filters
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const float *bias,
    float *out_layer,       //o/p to this function
    //unsigned char *out_layer,     // float or char?
    //const int out_no_of_images,
    //const int out_channels,       // same as no. of filters
    const int out_height,           //o/p to this function
    const int out_width             //o/p to this function
) {

#if 1
    //unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) thread_qty = 1;

    //Need to change this for latency optimization
    if (thread_qty > no_of_images) {
        thread_qty = no_of_images;
    }


    unsigned int loopCount = (no_of_images%thread_qty)==0 ?
                             no_of_images/thread_qty : (no_of_images/thread_qty)+1;
    //if(no_of_images > cpuVitualCores)
    //  bufferBucket = cpuVitualCores;

    unsigned long  size = (kernel_h*kernel_w*channels)*(out_height*out_width) *
                          no_of_images;
    float *data_col = (float *)malloc(size * sizeof(float));
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2D_ver4 Memory Error while allocating patch matrix");
        return;
    }

    //Need to optimize this
    //float *filterNew = filter;//transpose(filter, no_of_filter, kernel_h*kernel_w*channels);

    #pragma omp parallel num_threads(thread_qty)
    {
        for (int i=0; i<loopCount; i++) {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            if (threadOffset >= no_of_images) {
                break;
            }
            unsigned long bufferOffset = ((kernel_h*kernel_w*channels)*
                                          (out_height*out_width) * threadOffset);
            unsigned long inputOffset = (channels*height*width*threadOffset);
            //unsigned int bufferOffset = ((kernel_h*kernel_w*channels)*(out_height*out_width) * omp_get_thread_num());
            //unsigned int outBufferOffset = (no_of_filter* (out_height*out_width)* threadOffset);

            //#New Implementation with im2col and gemm function.
            //im2col_par(in_layer+(channels*height*width*i), channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, data_col + ((kernel_h*kernel_w*channels)*(out_height*out_width)) * threadOffset);

            //im2row is more efficient than im2col with NHWC
            //im2colNHWC(in_layer+(channels*height*width*threadOffset), channels, height, width, kernel_h, kernel_w, pad_h, pad_w, pad_h, pad_w, stride_h, stride_w, data_col + bufferOffset);
            im2rowNHWC(in_layer+inputOffset, channels, height, width, kernel_h, kernel_w,
                       pad_h, pad_w, pad_h, pad_w, stride_h, stride_w, data_col + bufferOffset);
        }

    }

    //AMD BLIS bases matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                out_height*out_width*no_of_images, no_of_filter, channels*kernel_h*kernel_w,
                1.0f,
                data_col, channels*kernel_h*kernel_w, filter, no_of_filter, 0.0f, out_layer,
                no_of_filter);

    #pragma omp parallel num_threads(thread_qty)
    {
        for (int i=0; i<loopCount; i++) {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            if (threadOffset >= no_of_images) {
                break;
            }
            unsigned long outBufferOffset = (no_of_filter* (out_height*out_width)*
                                             threadOffset);

            //Below Bias and activation code can be eliminated if not required
            //  //Functionally incorrect needto chenge the order for NHWC
            /*
               for(int r=0; r<no_of_filter; r++)
               {
               for(int m=0; m<out_height*out_width; m++)
               {
               out_layer[outBufferOffset + (r*(out_height*out_width)) + m]+= bias[r];
               }
               }
               */
            //sigmoid activation function

            //for(int r=0; r<no_of_filter*out_height*out_width; r++){
            //        out_layer[(no_of_filter*(out_height*out_width)*i) + r]=255.999f/(1+expf(-y[r]/256));
            //}
            //////////////////#pragma omp barrier
        }
    }

    //free(filter);
    free(data_col);

#else
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
void zenConvolution2D_ver5(
    //const unsigned char* in_layer,
    const float *in_layer,  // float or char input?
    const int no_of_images,
    const int channels,
    const int height,
    const int width,
    const float *filter,
    const int no_of_filter,
    //const int channels,           //no. of channels is same as no. of channels filters
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const float *bias,
    float *out_layer,       //o/p to this function
    //unsigned char *out_layer,     // float or char?
    //const int out_no_of_images,
    //const int out_channels,       // same as no. of filters
    const int out_height,           //o/p to this function
    const int out_width             //o/p to this function
) {

#if 1
    //unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) thread_qty = 1;

    unsigned int loopCount = (no_of_images%thread_qty)==0 ?
                             no_of_images/thread_qty : (no_of_images/thread_qty)+1;
    //if(no_of_images > cpuVitualCores)
    //  bufferBucket = cpuVitualCores;

    unsigned long  size = (kernel_h*kernel_w*channels)*(out_height*out_width) *
                          thread_qty;
    float *data_col = (float *)malloc(size * sizeof(float));
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2D_ver5 Memory Error while allocating patch matrix");
        return;
    }

    //Need to optimize
    //float *filterNew = filter;//transpose(filter, no_of_filter, kernel_h*kernel_w*channels);

    //Need to check if it make sence to free here
    //free((float *)filter);



    for (int i=0; i<loopCount; i++) {
        unsigned int outBatchSize = thread_qty;
        if (i==(loopCount-1) && (no_of_images%thread_qty)!=0) {
            outBatchSize = no_of_images - ((no_of_images/thread_qty) * thread_qty);
        }

        unsigned long outBufferOffset = (no_of_filter* (out_height*out_width)*
                                         thread_qty * i);
        #pragma omp parallel num_threads(thread_qty)
        {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            //outBufferOffset = (no_of_filter* (out_height*out_width)* tmpBufferSize * i);
            if (threadOffset < no_of_images) {
                //         break;
                unsigned long bufferOffset = ((kernel_h*kernel_w*channels)*
                                              (out_height*out_width) * omp_get_thread_num());
                unsigned long inputOffset = (channels*height*width*threadOffset);
                //unsigned int bufferOffset = ((kernel_h*kernel_w*channels)*(out_height*out_width) * omp_get_thread_num());



                //#New Implementation with im2col and gemm function.
                //im2col().....parallel version is also available
                //im2col_par(in_layer+(channels*height*width*i), channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, data_col + ((kernel_h*kernel_w*channels)*(out_height*out_width)) * threadOffset);

                //im2row is more efficient than im2col with NHWC
                //im2colNHWC(in_layer+(channels*height*width*threadOffset), channels, height, width, kernel_h, kernel_w, pad_h, pad_w, pad_h, pad_w, stride_h, stride_w, data_col + bufferOffset);
                im2rowNHWC(in_layer+inputOffset, channels, height, width, kernel_h, kernel_w,
                           pad_h, pad_w, pad_h, pad_w, stride_h, stride_w, data_col + bufferOffset);
            }
        }



        //AMD BLIS bases matrix multiplication
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    out_height*out_width*outBatchSize, no_of_filter, channels*kernel_h*kernel_w,
                    1.0f,
                    data_col, channels*kernel_h*kernel_w, filter, no_of_filter, 0.0f,
                    out_layer+outBufferOffset, no_of_filter);

        //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out_height*out_width, no_of_filter, channels*kernel_h*kernel_w, 1.0f,
        //                                                     data_col+bufferOffset, channels*kernel_h*kernel_w, filter, no_of_filter, 0.0f, out_layer+outBufferOffset, no_of_filter);

        #pragma omp parallel num_threads(thread_qty)
        {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            //outBufferOffset = (no_of_filter* (out_height*out_width)* tmpBufferSize * i * omp_get_thread_num());
            outBufferOffset = (no_of_filter* (out_height*out_width)* threadOffset);
            if (threadOffset < no_of_images) {
                //         break;

                //Below Bias and activation code can be eliminated if not required
                //  //Functionally incorrect needto chenge the order for NHWC
                /*
                   for(int r=0; r<no_of_filter; r++)
                   {
                   for(int m=0; m<out_height*out_width; m++)
                   {
                   out_layer[outBufferOffset + (r*(out_height*out_width)) + m]+= bias[r];
                   }
                   }
                   */
                //sigmoid activation function

                //for(int r=0; r<no_of_filter*out_height*out_width; r++){
                //       out_layer[(no_of_filter*(out_height*out_width)*i) + r]=255.999f/(1+expf(-y[r]/256));
                //}
                //////////////////#pragma omp barrier
            }
        }
    }
    //free(filter);
    free(data_col);

#else
#endif
}



//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner for im2row and bias operation
//For gemm, OMP take care of the parallelism with serial call to gemm from BLIS
//Parallel smaller gemm can also be invoked by changing #if def
//We call it mix of threading inside and outside conv operation
void zenConvolution2D_ver6(
    //const unsigned char* in_layer,
    const float *in_layer,  // float or char input?
    const int no_of_images,
    const int channels,
    const int height,
    const int width,
    const float *filter,
    const int no_of_filter,
    //const int channels,           //no. of channels is same as no. of channels filters
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const float *bias,
    float *out_layer,       //o/p to this function
    //unsigned char *out_layer,     // float or char?
    //const int out_no_of_images,
    //const int out_channels,       // same as no. of filters
    const int out_height,           //o/p to this function
    const int out_width             //o/p to this function
) {

#if 1
    //unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) thread_qty = 1;

    unsigned int loopCount = (no_of_images%thread_qty)==0 ?
                             no_of_images/thread_qty : (no_of_images/thread_qty)+1;
    //if(no_of_images > cpuVitualCores)
    //  bufferBucket = cpuVitualCores;

    unsigned long  size = (kernel_h*kernel_w*channels)*(out_height*out_width) *
                          thread_qty;
    float *data_col = (float *)malloc(size * sizeof(float));
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2D_ver6 Memory Error while allocating patch matrix");
        return;
    }

    //Need to optimize
    //float *filterNew = filter;//transpose(filter, no_of_filter, kernel_h*kernel_w*channels);

    for (int i=0; i<loopCount; i++) {
        unsigned int outBatchSize = thread_qty;
        if (i==(loopCount-1) && (no_of_images%thread_qty)!=0) {
            outBatchSize = no_of_images - ((no_of_images/thread_qty) * thread_qty);
        }

        unsigned long outBufferOffset = (no_of_filter* (out_height*out_width)*
                                         thread_qty * i);
        #pragma omp parallel num_threads(thread_qty)
        {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            //outBufferOffset = (no_of_filter* (out_height*out_width)* tmpBufferSize * i);
            if (threadOffset < no_of_images) {
                //         break;
                unsigned long bufferOffset = ((kernel_h*kernel_w*channels)*
                                              (out_height*out_width) * omp_get_thread_num());
                unsigned long inputOffset = (channels*height*width*threadOffset);
                //unsigned int bufferOffset = ((kernel_h*kernel_w*channels)*(out_height*out_width) * omp_get_thread_num());



                //#New Implementation with im2col and gemm function.
                //im2col().....parallel version is also available
                //im2col_par(in_layer+(channels*height*width*i), channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, data_col + ((kernel_h*kernel_w*channels)*(out_height*out_width)) * threadOffset);

                //im2row is more efficient than im2col with NHWC
                //im2colNHWC(in_layer+(channels*height*width*threadOffset), channels, height, width, kernel_h, kernel_w, pad_h, pad_w, pad_h, pad_w, stride_h, stride_w, data_col + bufferOffset);
                im2rowNHWC(in_layer+inputOffset, channels, height, width, kernel_h, kernel_w,
                           pad_h, pad_w, pad_h, pad_w, stride_h, stride_w, data_col + bufferOffset);
            }
        }

        //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out_height*out_width*outBatchSize, no_of_filter, channels*kernel_h*kernel_w, 1.0f,
        //              data_col, channels*kernel_h*kernel_w, filter, no_of_filter, 0.0f, out_layer+outBufferOffset, no_of_filter);
#if 1
        int workSize = (out_height*out_width*outBatchSize);
        int rowloop = ((out_height*out_width*outBatchSize)%(thread_qty*workSize)==0)?
                      (out_height*out_width*outBatchSize)/(thread_qty*workSize):((
                                  out_height*out_width*outBatchSize)/(thread_qty*workSize))+1;
        #pragma omp parallel num_threads(thread_qty)
        {
            //AMD BLIS bases matrix multiplication
            for (int j=0; j<rowloop; j++) {
                int workSizeNew = workSize;
                int threadOffset = (omp_get_thread_num()*workSize)+ (j*(thread_qty*workSize));
                if (threadOffset >= (out_height*out_width*outBatchSize)) {
                    break;
                }
                if (j == (rowloop-1) &&
                        ((out_height*out_width*outBatchSize) - threadOffset) < workSize) {
                    workSizeNew = (out_height*out_width*outBatchSize) - threadOffset;
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, workSizeNew,
                            no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                            data_col+(threadOffset*(channels*kernel_h*kernel_w)),
                            channels*kernel_h*kernel_w, filter, no_of_filter, 0.0f,
                            out_layer+outBufferOffset+(threadOffset*no_of_filter), no_of_filter);
            }
        }
#else
        int workSize = (out_height*out_width*outBatchSize);
        int rowloop = ((out_height*out_width*outBatchSize)%workSize)==0?
                      (out_height*out_width*outBatchSize)/(workSize):((
                                  out_height*out_width*outBatchSize)/(workSize))+1;
        for (int j=0; j<rowloop; j++) {
            int workSizeNew = workSize;

            if (j == (rowloop -1)) {
                workSizeNew = (out_height*out_width*outBatchSize) - (j*workSize);
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, workSizeNew,
                        no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                        data_col+(j*workSize*(channels*kernel_h*kernel_w)), channels*kernel_h*kernel_w,
                        filter, no_of_filter, 0.0f, out_layer+outBufferOffset+(j*workSize*no_of_filter),
                        no_of_filter);
        }



#endif
        #pragma omp parallel num_threads(thread_qty)
        {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            //outBufferOffset = (no_of_filter* (out_height*out_width)* tmpBufferSize * i * omp_get_thread_num());
            outBufferOffset = (no_of_filter* (out_height*out_width)* threadOffset);
            if (threadOffset < no_of_images) {
                //         break;

                //Below Bias and activation code can be eliminated if not required
                //Functionally incorrect needto chenge the order for NHWC
                /*
                   for(int r=0; r<no_of_filter; r++)
                   {
                   for(int m=0; m<out_height*out_width; m++)
                   {
                   out_layer[outBufferOffset + (r*(out_height*out_width)) + m]+= bias[r];
                   }
                   }
                   */
                //sigmoid activation function

                //for(int r=0; r<no_of_filter*out_height*out_width; r++){
                //       out_layer[(no_of_filter*(out_height*out_width)*i) + r]=255.999f/(1+expf(-y[r]/256));
                //}
                //////////////////#pragma omp barrier
            }
        }
    }
    //free(filter);
    free(data_col);

#else
#endif
}



//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner for im2row, gemm and bias operation
//internally these oprtaion runs sequntially....we call it threading outside operation
void zenConvolution2DbaseRef(
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
    const float *scale
) {

#if 0
    unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) thread_qty = 1;

    struct timeval start, end;
    gettimeofday(&start, 0);

#endif
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    //Need to change this for latency optimization
    if (thread_qty > no_of_images) {
        thread_qty = no_of_images;
    }

    unsigned long data_col_size = ((unsigned long)(kernel_h*kernel_w*channels)*
                                   (out_height*out_width)*sizeof(float)*thread_qty);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = (float *)zendnn_aligned_alloc(ALIGNED_OFFSET, data_col_size);

    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DbaseRef Memory Error while allocating patch matrix");
        return;
    }
    //Running Ref version as single threaded
    thread_qty = 1;
    #pragma omp parallel num_threads(thread_qty)
    {
        unsigned int loopCount = (no_of_images%thread_qty)==0 ?
                                 no_of_images/thread_qty : (no_of_images/thread_qty)+1;
        for (int i=0; i<loopCount; i++) {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            if (threadOffset >= no_of_images) {
                break;
            }
            unsigned long inputOffset = ((unsigned long)channels*height*width*threadOffset);
            unsigned long patchInputOffset = ((unsigned long)(kernel_h*kernel_w*channels)*
                                              (out_height*out_width) * omp_get_thread_num());
            unsigned long outputOffset = ((unsigned long)no_of_filter*
                                          (out_height*out_width)* threadOffset);

            //im2row is more efficient than im2col with NHWC
            im2rowNHWC(in_layer+inputOffset, channels, height, width, kernel_h, kernel_w,
                       pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, data_col + patchInputOffset);

            //AMD BLIS bases matrix multiplication
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out_height*out_width,
                        no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                        data_col+patchInputOffset, channels*kernel_h*kernel_w, filter, no_of_filter,
                        0.0f, out_layer+outputOffset, no_of_filter);

            if (bias && !relu) {
                //Below Bias and activation code can be eliminated if not required
                for (int m=0; m<out_height*out_width; m++)
                    for (int r=0; r<no_of_filter; r++) {
                        if (scale) {
                            out_layer[outputOffset +(m*(no_of_filter)) + r]= (out_layer[outputOffset +(m*
                                    (no_of_filter)) + r]*scale[r]) + bias[r];
                        }
                        else {
                            out_layer[outputOffset +(m*(no_of_filter)) + r]+= bias[r];
                        }

                        //out_layer[outputOffset+ (m*(no_of_filter)) + r]+= bias[r];
                    }
            }
            if (bias && relu) {
                //Below Bias and activation code can be eliminated if not required
                for (int m=0; m<out_height*out_width; m++)
                    for (int r=0; r<no_of_filter; r++) {
                        if (scale) {
                            out_layer[outputOffset +(m*(no_of_filter)) + r]= (out_layer[outputOffset +(m*
                                    (no_of_filter)) + r]*scale[r]) + bias[r];
                        }
                        else {
                            out_layer[outputOffset +(m*(no_of_filter)) + r]+= bias[r];
                        }
                        if (out_layer[outputOffset +(m*(no_of_filter)) + r] < 0) {
                            out_layer[outputOffset +(m*(no_of_filter)) + r] = 0;
                        }

                        //out_layer[outputOffset+ (m*(no_of_filter)) + r]+= bias[r];
                        //if (out_layer[outputOffset + (m*(no_of_filter)) + r] < 0) {
                        //    out_layer[outputOffset + (m*(no_of_filter)) + r] = 0;
                        // }
                    }
            }
        }
    }
    free(data_col);
#if 0
    gettimeofday(&end, 0);
    float elapsed;
    elapsed = timedifference_msec(start, end);
    zendnnInfo(ZENDNN_PROFLOG, "zenConvolution2D_best, no_of_images=", no_of_images,
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
//internally these oprtaion runs sequntially....we call it threading outside operation
//Divide gemm calls for each i/p frame to smaller chunks
void zenConvolution2D_SmallGemm(
    //const unsigned char* in_layer,
    const float *in_layer,  // float or char input?
    const int no_of_images,
    const int channels,
    const int height,
    const int width,
    const float *filter,
    const int no_of_filter,
    //const int channels,           //no. of channels is same as no. of channels filters
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const float *bias,
    float *out_layer,       //o/p to this function
    //unsigned char *out_layer,     // float or char?
    //const int out_no_of_images,
    //const int out_channels,       // same as no. of filters
    const int out_height,           //o/p to this function
    const int out_width             //o/p to this function
) {

#if 1
    //unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) thread_qty = 1;

    //Need to change this for latency optimization
    if (thread_qty > no_of_images) {
        thread_qty = no_of_images;
    }


    unsigned long  size = (kernel_h*kernel_w*channels)*(out_height*out_width) *
                          thread_qty;
    float *data_col = (float *)malloc(size * sizeof(float));
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2D_SmallGemm Memory Error while allocating patch matrix");
        return;
    }

    //Need to optimize this
    //float *filterNew = filter;//transpose(filter, no_of_filter, kernel_h*kernel_w*channels);


    #pragma omp parallel num_threads(thread_qty)
    {
        unsigned int loopCount = (no_of_images%thread_qty)==0 ?
                                 no_of_images/thread_qty : (no_of_images/thread_qty)+1;
        for (int i=0; i<loopCount; i++) {
            int threadOffset = omp_get_thread_num()+ (i*thread_qty);
            if (threadOffset >= no_of_images) {
                break;
            }
            unsigned long inputOffset = (channels*height*width*threadOffset);
            unsigned long patchInputOffset = ((kernel_h*kernel_w*channels)*
                                              (out_height*out_width) * omp_get_thread_num());
            unsigned long  outputOffset = (no_of_filter* (out_height*out_width)*
                                           threadOffset);


            //#New Implementation with im2col and gemm function.
            //im2col_par(in_layer+(channels*height*width*i), channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, data_col + ((kernel_h*kernel_w*channels)*(out_height*out_width)) * threadOffset);

            //im2row is more efficient than im2col with NHWC
            im2rowNHWC(in_layer+inputOffset, channels, height, width, kernel_h, kernel_w,
                       pad_h, pad_w, pad_h, pad_w, stride_h, stride_w, data_col + patchInputOffset);


            //AMD BLIS bases matrix multiplication
            //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out_height*out_width, no_of_filter, channels*kernel_h*kernel_w, 1.0f,
            //                                                       data_col+patchInputOffset, channels*kernel_h*kernel_w, filter, no_of_filter, 0.0f, out_layer+outputOffset, no_of_filter);
            //int workSize = no_of_filter*4096;
            int workSize = (out_height*out_width)/2;
            int rowloop = ((out_height*out_width)%workSize)==0?(out_height*out_width)/
                          (workSize):((out_height*out_width)/(workSize))+1;
            for (int j=0; j<rowloop; j++) {
                int workSizeNew = workSize;
                if (j == (rowloop -1)) {
                    workSizeNew = (out_height*out_width) - (j*workSize);
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, workSizeNew,
                            no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                            data_col+patchInputOffset+(j*workSize*(channels*kernel_h*kernel_w)),
                            channels*kernel_h*kernel_w, filter, no_of_filter, 0.0f,
                            out_layer+outputOffset+(j*workSize*no_of_filter), no_of_filter);
            }


            //Below Bias and activation code can be eliminated if not required
            /*Functionally incorrect need to change the order for NHWC
              for(int r=0; r<no_of_filter; r++)
              {
              for(int m=0; m<out_height*out_width; m++)
              {
              out_layer[outputOffset+ (r*(out_height*out_width)) + m]+= bias[r];
              }
              }
              */
            //sigmoid activation function
            /*
               for(int r=0; r<no_of_filter*out_height*out_width; r++){
               out_layer[(no_of_filter*(out_height*out_width)*i) + r]=255.999f/(1+expf(-y[r]/256));
               } */
            //////////////////#pragma omp barrier
        }
    }
    //free(filter);
    free(data_col);

#else
#endif
}



//This implementation is based on im2row and gemm(BLIS) where im2row is performed on all the input
//images/featureMap followed by gemm call to blis which computes the feature map for all the images
//and then add bias value on it.
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner for im2row, gemm and bias operation
//internally these oprtaion runs sequntially....we call it threading outside operation
//Merge two i/p image gemm callsinto bigger one
void zenConvolution2D_BigGemm(
    //const unsigned char* in_layer,
    const float *in_layer,  // float or char input?
    const int no_of_images,
    const int channels,
    const int height,
    const int width,
    const float *filter,
    const int no_of_filter,
    //const int channels,           //no. of channels is same as no. of channels filters
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const float *bias,
    float *out_layer,       //o/p to this function
    //unsigned char *out_layer,     // float or char?
    //const int out_no_of_images,
    //const int out_channels,       // same as no. of filters
    const int out_height,           //o/p to this function
    const int out_width             //o/p to this function
) {

#if 1
    //unsigned short cpuVitualCores = get_nprocs();
    unsigned int thread_qty = zendnn_getenv_int("OMP_NUM_THREADS");
    if (thread_qty == 0) thread_qty = 1;

    //Need to change this for latency optimization
    if (thread_qty > no_of_images) {
        thread_qty = no_of_images;
    }


    unsigned long  size = (kernel_h*kernel_w*channels)*(out_height*out_width) *
                          no_of_images;
    float *data_col = (float *)malloc(size * sizeof(float));
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2D_BigGemm Memory Error while allocating patch matrix");
        return;
    }

    //Need to optimize this
    //float *filterNew = filter;//transpose(filter, no_of_filter, kernel_h*kernel_w*channels);


    #pragma omp parallel num_threads(thread_qty)
    {
        //unsigned int loopCount = (no_of_images%thread_qty)==0 ? no_of_images/thread_qty : (no_of_images/thread_qty)+1;
        unsigned int loopCount = (no_of_images%thread_qty)==0 ? 1 : 2;
        unsigned int gemmBatchSize = no_of_images/thread_qty;
        unsigned int batchSize = gemmBatchSize;
        for (int i=0; i<loopCount; i++) {
            int flag = 0;

            if (i>0) {
                gemmBatchSize = 1;
            }
            for (int j=0; j<gemmBatchSize; j++) {

                unsigned long threadOffset = (i*thread_qty*batchSize) +
                                             (omp_get_thread_num()*gemmBatchSize) + j;
                if (threadOffset >= no_of_images) {
                    flag = 1;
                    break;
                }
                unsigned long inputOffset = (channels*height*width*threadOffset);
                unsigned long patchInputOffset = ((kernel_h*kernel_w*channels)*
                                                  (out_height*out_width) * threadOffset);


                //#New Implementation with im2col and gemm function.
                //im2col_par(in_layer+(channels*height*width*i), channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, data_col + ((kernel_h*kernel_w*channels)*(out_height*out_width)) * threadOffset);

                //im2row is more efficient than im2col with NHWC
                im2rowNHWC(in_layer+inputOffset, channels, height, width, kernel_h, kernel_w,
                           pad_h, pad_w, pad_h, pad_w, stride_h, stride_w, data_col + patchInputOffset);

            }
            unsigned long outputOffset = (no_of_filter* (out_height*out_width) * ((
                                              i*thread_qty*batchSize) + (omp_get_thread_num()*gemmBatchSize)));
            unsigned long data_col_offset = ((i*thread_qty*batchSize) +
                                             (omp_get_thread_num()*gemmBatchSize)) * (kernel_h*kernel_w*channels)*
                                            (out_height*out_width);
            //AMD BLIS bases matrix multiplication
#if 0
            if (!flag) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            out_height*out_width*gemmBatchSize, no_of_filter, channels*kernel_h*kernel_w,
                            1.0f,
                            data_col+data_col_offset, channels*kernel_h*kernel_w, filter, no_of_filter,
                            0.0f, out_layer+outputOffset, no_of_filter);
            }
#else
            //int workSize = no_of_filter*4096;
            if (!flag) {
                int workSize = (out_height*out_width*gemmBatchSize)/32;
                int rowloop = ((out_height*out_width*gemmBatchSize)%workSize)==0?
                              (out_height*out_width*gemmBatchSize)/(workSize):((
                                          out_height*out_width*gemmBatchSize)/(workSize))+1;
                for (int j=0; j<rowloop; j++) {
                    int workSizeNew = workSize;
                    if (j == (rowloop -1)) {
                        workSizeNew = (out_height*out_width*gemmBatchSize) - (j*workSize);
                    }
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, workSizeNew,
                                no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                                data_col+data_col_offset+(j*workSize*(channels*kernel_h*kernel_w)),
                                channels*kernel_h*kernel_w, filter, no_of_filter, 0.0f,
                                out_layer+outputOffset+(j*workSize*no_of_filter), no_of_filter);
                }
            }

#endif
            for (int j=0; j<gemmBatchSize; j++) {
                int threadOffset = (i*thread_qty*batchSize) + (omp_get_thread_num()
                                   *gemmBatchSize) + j;
                if (threadOffset >= no_of_images) {
                    break;
                }
                unsigned long biasOffset = no_of_filter* (out_height*out_width) * threadOffset;
                //Below Bias and activation code can be eliminated if not required
                /*
                 * Functionally incorrect need to change the order for NHWC
                 for(int r=0; r<no_of_filter; r++)
                 {
                 for(int m=0; m<out_height*out_width; m++)
                 {
                 out_layer[biasOffset + (r*(out_height*out_width)) + m]+= bias[r];
                 }
                 }
                 */
            }
            //sigmoid activation function
            /*
               for(int r=0; r<no_of_filter*out_height*out_width; r++){
               out_layer[(no_of_filter*(out_height*out_width)*i) + r]=255.999f/(1+expf(-y[r]/256));
               } */
            //////////////////#pragma omp barrier
        }
    }
    //free(filter);
    free(data_col);

#else
#endif
}


//An umbrella C++ interface for zendnn convolution
void zenConvolution2DgemmRef(
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
    const float *scale
) {

    //In future this will be part of zendnn initialization
    zendnnEnv zenEnvObj = readEnv();

#ifdef _WIN32
    auto start = std::chrono::high_resolution_clock::now();
#else
    struct timeval start, end;
    gettimeofday(&start, 0);
#endif


    zenConvolution2DbaseRef(zenEnvObj, in_layer, batchsize, channels, height, width,
                            filter, no_of_filter,
                            kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                            out_layer, out_height, out_width, relu, scale);

float elapsed;
#ifdef _WIN32
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> difference = end - start;
    elapsed = difference.count();
#else
    gettimeofday(&end, 0);
    elapsed = timedifference_msec(start, end);
#endif
    zendnnInfo(ZENDNN_PROFLOG, "zenConvolution2DbaseRef, no_of_images=", batchsize,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w,
               " Time=", elapsed, "ms");
}

void zenConvolution2DRef(
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
    const int out_width
) {

    //Need to perforam other checks...eg. for all input dimansions
    if ((in_layer == NULL)|| (filter == NULL) || (out_layer == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2D Memory is not defined for in_layer or filter or out_layer");
        return;
    }
    //int pad_t,pad_l,pad_b,pad_r;
    //compute_padding(pad_h,pad_w,&pad_t,&pad_l,&pad_b,&pad_r);
    zenConvolution2DgemmRef(in_layer, batchsize, channels, height, width, filter,
                            no_of_filter,
                            kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, NULL,
                            out_layer, out_height, out_width, 0, NULL);


}

void zenConvolution2DwithBiasRef(
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
    const int out_width
) {
    //Need to perforam other checks...eg. for all input dimansions
    if ((in_layer == NULL)|| (filter == NULL) || (out_layer == NULL) ||
            (bias == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DwithBias Memory is not defined for in_layer or filter or out_layer");
        return;
    }
    //int pad_t,pad_l,pad_b,pad_r;
    //compute_padding(pad_h,pad_w,&pad_t,&pad_l,&pad_b,&pad_r);
    zenConvolution2DgemmRef(in_layer, batchsize, channels, height, width, filter,
                            no_of_filter,
                            kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                            out_layer, out_height, out_width, 0, NULL);
}

void zenConvolution2DwithBiasReluRef(
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
    const int out_width
) {

    //Need to perforam other checks...eg. for all input dimansions
    if ((in_layer == NULL)|| (filter == NULL) || (out_layer == NULL) ||
            (bias == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DwithBiasRelu Memory is not defined for in_layer or filter or out_layer");
        return;
    }
    //int pad_t,pad_l,pad_b,pad_r;
    //compute_padding(pad_h,pad_w,&pad_t,&pad_l,&pad_b,&pad_r);
    zenConvolution2DgemmRef(in_layer, batchsize, channels, height, width, filter,
                            no_of_filter,
                            kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                            out_layer, out_height, out_width, 1, NULL);
}


void zenConvolution2DwithBatchNormRef(
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
    const int out_width
) {

    //Need to perforam other checks...eg. for all input dimansions
    if ((in_layer == NULL)|| (filter == NULL) || (out_layer == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2DwithBatchNorm Memory is not defined for in_layer or filter or out_layer");
        return;
    }
    //int pad_t,pad_l,pad_b,pad_r;
    //compute_padding(pad_h,pad_w,&pad_t,&pad_l,&pad_b,&pad_r);
    float *bias = (float *)malloc(sizeof(float)*no_of_filter);
    for (int r=0; r <no_of_filter; r++) {
        bias[r] = offset[r]-(scale[r]*mean[r]);
    }

    zenConvolution2DgemmRef(in_layer, batchsize, channels, height, width, filter,
                            no_of_filter,
                            kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                            out_layer, out_height, out_width, 0, scale);
    free(bias);
}

void zenConvolution2DwithBatchNormReluRef(
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
    const int out_width
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
    for (int r=0; r <no_of_filter; r++) {
        bias[r] = offset[r]-(scale[r]*mean[r]);
    }

    zenConvolution2DgemmRef(in_layer, batchsize, channels, height, width, filter,
                            no_of_filter,
                            kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias,
                            out_layer, out_height, out_width, 1, scale);

    free(bias);
}
