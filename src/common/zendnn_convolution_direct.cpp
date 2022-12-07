/*******************************************************************************
* Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include "common/zendnn_private.hpp"
#include <omp.h>
#include <cblas.h>
#include <time.h>
#include "zendnn_logging.hpp"
#include "zendnn_helper.hpp"

#define ALIGNED_OFFSET          64

using namespace zendnn;

//This implementation is based on direct convolution and sgemv(BLIS)
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner, sgemv and bias operation
//internally these oprtaion runs sequntially....we call it threading outside operation
void zenConvolution2D_direct(
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
    const float *scale,
    const float *elementwise_input

) {

    zendnnInfo(ZENDNN_PROFLOG, "zenConvolution2D_direct, no_of_images=",
               no_of_images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w);

    printf(" CblasRowMajor CblasTrans M, N, LDA \t%d\t%d\t%d\n",
           channels*kernel_h*kernel_w, no_of_filter, no_of_filter);
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    //Need to change this for latency optimization
    if (thread_qty > no_of_images) {
        thread_qty = no_of_images;
    }

    unsigned long data_col_size = ((kernel_h*kernel_w*channels)*sizeof(
                                       float)*thread_qty);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = (float *)zendnn_aligned_alloc(ALIGNED_OFFSET, data_col_size);
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2D_direct Memory Error while allocating patch matrix");
        return;
    }

    //Need to optimize this
    float *filterNew = (float *)
                       filter;//transpose(filter, no_of_filter, kernel_h*kernel_w*channels);
    //float *filterNew = (float *)transpose(filter, no_of_filter, kernel_h*kernel_w*channels);

    int height_col =
        out_height; //(height + pad_h + pad_w - kernel_h) / stride_h + 1;
    int width_col = out_width; //(width + pad_h + pad_w - kernel_w) / stride_w + 1;
    //return;
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
            unsigned long patchInputOffset = ((kernel_h*kernel_w*channels) *
                                              omp_get_thread_num());
            unsigned long outputOffset = ((unsigned long)no_of_filter*
                                          (out_height*out_width)* threadOffset);

            float *data_col_tmp = data_col + patchInputOffset;
            unsigned int data_col_offset = 0;
            unsigned int out_count = 0;

            int h = 0;
            int h_pad = -pad_t;
            for (h = 0; h < height_col; ++h) {
                int w_pad = -pad_l;
                //#pragma omp parallel for num_threads(4)
                for (int w = 0; w < width_col; ++w) {
                    data_col_offset = 0;
                    for (int ih = h_pad; ih < h_pad + kernel_h; ++ih) {
                        for (int iw = w_pad; iw < w_pad + kernel_w; ++iw) {
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int offset = (inputOffset) + (ih * width + iw) * channels;
                                for (int k = 0; k<channels; k++) {
                                    data_col_tmp[data_col_offset + k] = in_layer[offset + k];
                                }
                            }
                            else {
                                // This should be simply padded with zero.
                                for (int k = 0; k<channels; k++) {
                                    data_col_tmp[data_col_offset + k] = 0;
                                }
                            }
                            data_col_offset += channels;
                        }
                    }
                    w_pad += stride_w;

                    //SGEMV call
                    //filter is NHWC...transposing above

                    //Working if filter is NCHW
                    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, no_of_filter, 1, channels*kernel_h*kernel_w, 1.0f,
                    //            filterNew, channels*kernel_h*kernel_w, data_col_tmp, 1, 0.0f, out_layer + outputOffset + (no_of_filter*out_count), 1);

                    //Working if filter is NHWC
                    //cblas_sgemv(CblasRowMajor, CblasNoTrans, no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                    //        filterNew, channels*kernel_h*kernel_w, data_col_tmp, 1, 0.0f, out_layer + outputOffset + (no_of_filter*out_count), 1);

                    //Working if filter is HWCN
                    cblas_sgemv(CblasRowMajor, CblasTrans, channels*kernel_h*kernel_w, no_of_filter,
                                1.0f,
                                filterNew, no_of_filter, data_col_tmp, 1, 0.0f,
                                out_layer + outputOffset + (no_of_filter*out_count), 1);

                    //working if filter is HWCN
                    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, no_of_filter, channels*kernel_h*kernel_w, 1.0f,
                    //         data_col_tmp, channels*kernel_h*kernel_w, filterNew, no_of_filter, 0.0f, out_layer + outputOffset + (no_of_filter*out_count), no_of_filter);
                    //data_col_tmp += data_col_offset;
                    out_count++;
                }

                h_pad += stride_h;
            }
            zenPostOps(zenEnvObj, out_layer, elementwise_input, out_height, out_width,
                       no_of_filter, no_of_filter,
                       outputOffset, bias,
                       relu, 0, scale, 1,0,0,no_of_images);
        }
    }
    free(data_col);

}

//This implementation is based on direct convolution and sgemv(BLIS)
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner, sgemv and bias operation
//internally these oprtaion runs sequntially....we call it threading outside operation
void zenConvolution2D_directVer2(
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
    const float *scale,
    const float *elementwise_input
) {

    zendnnInfo(ZENDNN_PROFLOG, "zenConvolution2D_directVer2, no_of_images=",
               no_of_images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w);

    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    //Need to change this for latency optimization
    if (thread_qty > no_of_images) {
        thread_qty = no_of_images;
    }

    //unsigned long data_col_size = ((kernel_h*kernel_w*channels)*sizeof(float)*thread_qty);
    unsigned long data_col_size = (channels)*sizeof(float)*thread_qty;
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = (float *)zendnn_aligned_alloc(ALIGNED_OFFSET, data_col_size);
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2D_directVer2 Memory Error while allocating patch matrix");
        return;
    }

    //Need to optimize this
    //float *filterNew = (float *)filter;//transpose(filter, no_of_filter, kernel_h*kernel_w*channels);
    float *filterNew = transpose(filter, kernel_h*kernel_w*channels, no_of_filter);
    //float *directOut = (float*) malloc(no_of_filter * sizeof(float)*thread_qty);


    int height_col =
        out_height; //(height + pad_h + pad_w - kernel_h) / stride_h + 1;
    int width_col = out_width; //(width + pad_h + pad_w - kernel_w) / stride_w + 1;
    //return;
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
            unsigned long patchInputOffset = ((channels) * omp_get_thread_num());
            //unsigned long directOutOffset = ((no_of_filter) * omp_get_thread_num());
            unsigned long outputOffset = (no_of_filter* (out_height*out_width)*
                                          threadOffset);

            unsigned int data_col_offset = 0;
            unsigned int out_count = 0;
            float temp_out = 0;

            int h = 0;
            int h_pad = -pad_t;
            for (h = 0; h < height_col; ++h) {
                int w_pad = -pad_l;
                for (int w = 0; w < width_col; ++w) {
                    data_col_offset = 0;
                    //memset(directOut+directOutOffset, 0, sizeof(float)*no_of_filter);
                    memset(&out_layer[outputOffset + (no_of_filter*out_count)], 0,
                           sizeof(float)*no_of_filter);

                    for (int ih = h_pad; ih < h_pad + kernel_h; ++ih) {
                        for (int iw = w_pad; iw < w_pad + kernel_w; ++iw) {
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                for (int f=0; f<no_of_filter; f++) {
                                    //cblas_sgemv(CblasRowMajor, CblasNoTrans, 1, channels, 1.0f,
                                    //filterNew + (data_col_offset*channels) + (f*channels*kernel_h*kernel_w),
                                    //                  channels, in_layer + (inputOffset) + (ih * width + iw) * channels,
                                    //                        1, 0.0f, &temp_out, 1);

                                    temp_out = cblas_sdot(channels,
                                                          filterNew+(data_col_offset*channels) + (f*channels*kernel_h*kernel_w), 1,
                                                          in_layer + (inputOffset) + (ih * width + iw) * channels, 1);

                                    //directOut[directOutOffset + f] += temp_out;
                                    out_layer[outputOffset + (no_of_filter*out_count) + f]  += temp_out;
                                }
                            }
                            /*
                                        else {
                                            // This should be simply padded with zero.
                                            memset(data_col_tmp, 0, sizeof(float) * channels);
                                            for (int f=0; f<no_of_filter; f++) {
                                                cblas_sgemv(CblasRowMajor, CblasNoTrans, 1, channels, 1.0f,
                                                            filterNew + (data_col_offset*channels) + (f*channels*kernel_h*kernel_w), channels,
                                                            data_col_tmp, 1, 0.0f, &temp_out, 1);
                                                directOut[f] += temp_out;
                                            }

                                        }
                            */

                            data_col_offset++;
                        }

                    }
                    w_pad += stride_w;
                    //memcpy(out_layer + outputOffset + (no_of_filter*out_count), directOut + directOutOffset, sizeof(float)* no_of_filter);
                    out_count++;
                }

                h_pad += stride_h;
            }
            zenPostOps(zenEnvObj, out_layer, elementwise_input, out_height, out_width,
                       no_of_filter, no_of_filter,
                       outputOffset, bias,
                       relu, 0, scale, 1,0,0,no_of_images);
        }
    }
    //free(directOut);
    free(data_col);

}


//This implementation is based on direct convolution and sgemv(BLIS)
//I/p and o/p format will be NHWC and filter format is HWCN
//Multi thread parallization happen at OMP level in embarrassingly parallel manner, sgemv and bias operation
//internally these oprtaion runs sequntially....we call it threading outside operation
void zenConvolution2D_directVer3(
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
    const float *scale,
    const float *elementwise_input

) {

    zendnnInfo(ZENDNN_PROFLOG, "zenConvolution2D_directiVer3, no_of_images=",
               no_of_images,
               " channels=", channels, " height=", height, " width=", width,
               " no_of_filter=", no_of_filter, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_l=", pad_l,
               " pad_b=", pad_b, " pad_r=", pad_r,
               " stride_h=", stride_h, " stride_w=",stride_w);

    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    //Need to change this for latency optimization
    if (thread_qty > no_of_images) {
        thread_qty = no_of_images;
    }

    //unsigned long data_col_size = ((kernel_h*kernel_w*channels)*sizeof(float)*thread_qty);
    unsigned long data_col_size = (channels)*sizeof(float)*thread_qty;
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size :
                    (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = (float *)zendnn_aligned_alloc(ALIGNED_OFFSET, data_col_size);
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenConvolution2D_directVer3 Error while allocating patch matrix");
        return;
    }

    //Need to optimize this
    float *filterNew = (float *)
                       filter;//transpose(filter, no_of_filter, kernel_h*kernel_w*channels);

    int height_col =
        out_height; //(height + pad_h + pad_w - kernel_h) / stride_h + 1;
    int width_col = out_width;  //(width + pad_h + pad_w - kernel_w) / stride_w + 1;
    //return;
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
            unsigned long patchInputOffset = ((channels) * omp_get_thread_num());
            //unsigned long directOutOffset = ((no_of_filter) * omp_get_thread_num());
            unsigned long outputOffset = (no_of_filter* (out_height*out_width)*
                                          threadOffset);

            unsigned int data_col_offset = 0;
            unsigned int out_count = 0;
            float temp_out = 0;

            int h = 0;
            int h_pad = -pad_t;
            //#pragma omp parallel
            for (h = 0; h < height_col; ++h) {
                int w_pad = -pad_l;
                for (int w = 0; w < width_col; ++w) {
                    data_col_offset = 0;
                    memset(&out_layer[outputOffset + (no_of_filter*out_count)], 0,
                           sizeof(float)*no_of_filter);

                    for (int ih = h_pad; ih < h_pad + kernel_h; ++ih) {

                        if (0) { //(ih%kernel_h == 0) && (ih >= 0 && ih < height &&  w_pad>= 0 &&  w_pad< width) )
                            cblas_sgemv(CblasRowMajor, CblasTrans, channels*kernel_w, no_of_filter, 1.0f,
                                        filterNew + (data_col_offset*channels*no_of_filter), no_of_filter,
                                        in_layer + (inputOffset) + (ih * width + w_pad) * channels, 1, 1.0f,
                                        out_layer + outputOffset + (no_of_filter*out_count), 1);
                            data_col_offset += kernel_w;
                        }
                        else {
                            for (int iw = w_pad; iw < w_pad + kernel_w; ++iw) {
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    //memcpy(data_col + patchInputOffset, in_layer + (inputOffset) + (ih * width + iw) * channels, sizeof(float) * channels);

                                    cblas_sgemv(CblasRowMajor, CblasTrans, channels, no_of_filter, 1.0f,
                                                filterNew + (data_col_offset*channels*no_of_filter), no_of_filter,
                                                in_layer + (inputOffset) + (ih * width + iw) * channels, 1, 1.0f,
                                                out_layer + outputOffset + (no_of_filter*out_count), 1);
                                }
                                data_col_offset++;
                            }
                        }

                    }

                    w_pad += stride_w;
                    //memcpy(out_layer + outputOffset + (no_of_filter*out_count), directOut + directOutOffset, sizeof(float)* no_of_filter);
                    out_count++;
                }

                h_pad += stride_h;
            }
            zenPostOps(zenEnvObj, out_layer, elementwise_input, out_height, out_width,
                       no_of_filter, no_of_filter,
                       outputOffset, bias,
                       relu, 0, scale, 1,0,0,no_of_images);
        }
    }
    //free(directOut);
    free(data_col);
}
