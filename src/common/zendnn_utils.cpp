/*******************************************************************************
* Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <stdbool.h> // for padding_zone()
#include <zendnn_private.hpp>
#include "zendnn_logging.hpp"
#include "zendnn_helper.hpp"

using namespace zendnn;

// initialize memory pool static array for use by the kernels
// declared in zendnn_utils.hpp
ZenLibMemoryPool *ZenLibMemoryPool::zenLibMemPoolArr[ZEN_LIB_MEM_POOL_LIMIT] = {NULL};
int ZenLibMemoryPool::zenLibMemPoolCount = 0;


//Read env variables for zendnn
zendnnEnv readEnv() {
    zendnnEnv envObj;
    envObj.omp_num_threads = zendnn_getenv_int("OMP_NUM_THREADS", 1);
    if (getenv("ZEN_NUM_THREADS")) {
        envObj.zen_num_threads = atoi(getenv("ZEN_NUM_THREADS"));
        //Overriding OMP_NUM_THREADS if ZEN_NUM_THREADS is exported
        envObj.omp_num_threads = envObj.zen_num_threads;
    }

    //ZENDNN_BLOCKED_FORMAT is to enable/disable BLOCKED Format.
    envObj.zenBlockedFormat = zendnn_getenv_int("ZENDNN_BLOCKED_FORMAT", 0);

    //TODO: change ZENDNN_ENABLE_MEMPOOL to ZENDNN_ENABLE_TF_MEMPOOL
    //use ZENDNN_ENABLE_ONNX_MEMPOOL for ONNX
    //Possible values for ZENDNN_ENABLE_MEMPOOL
    // 0 (GAM-TPA disable)
    // 1 (Node level Memory Reuse)
    // 2 (Graph level Memory Reuse)
    envObj.zenEnableMemPool = zendnn_getenv_int("ZENDNN_ENABLE_MEMPOOL", 1);
    if(envObj.zenEnableMemPool < 0 || envObj.zenEnableMemPool > 2)
        envObj.zenEnableMemPool = 1;

    //TODO: Unified FWK and LIB mempool for next release
    envObj.zenLibMemPoolEnable = zendnn_getenv_int("ZENDNN_ENABLE_MEMPOOL", 1);

    //ZENDNN_INT8_SUPPORT is to enable/disable INT8 support
    envObj.zenINT8format = zendnn_getenv_int("ZENDNN_INT8_SUPPORT", 0);

    //ZENDNN_BLOCKED_NHWC is added to support NHWC data format for CONV DIRECT ALGO
    envObj.zenBlockedNHWC = zendnn_getenv_int("ZENDNN_NHWC_BLOCKED",0);

    //ZENDNN Library gives preference to NHWC-BLOCKED Format over BLOCKED Format.
    if (envObj.zenBlockedNHWC) {
        envObj.zenBlockedFormat=0;
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
    envObj.zenGEMMalgo = zendnn_getenv_int("ZENDNN_GEMM_ALGO", 3);
    if (envObj.zenGEMMalgo<=0 || envObj.zenGEMMalgo>3) {
        envObj.zenGEMMalgo = 3;
    }

    return envObj;
}

void compute_padding(const int image_h, const int image_w,
                     const int filter_h, const int filter_w,
                     const int stride_h, const int stride_w,
                     const char *padding,
                     int *pad_t,int *pad_l,int *pad_b, int *pad_r) {
    if (!strcmp(padding,"VALID")) {
        *pad_t = *pad_b = *pad_l = *pad_r = 0;
        return;
    }
    int total_pad_h, total_pad_w;
    int mod_h, mod_w;
    mod_h = image_h % stride_h;
    mod_w = image_w % stride_w;

    total_pad_h = std::max(filter_h - (mod_h == 0? stride_h: mod_h), 0);
    *pad_t = (total_pad_h / 2); // integer division equivalent to floor
    *pad_b = total_pad_h - *pad_t;

    total_pad_w = std::max(filter_w - (mod_w == 0? stride_w: mod_w), 0);
    *pad_l = (total_pad_w / 2); // integer division equivalent to floor
    *pad_r = total_pad_w - *pad_l;
}

// this is used in max_pooling op
// to determine if a data point is a part of original or padding data
bool padding_zone(int top_y, int top_x, int width_orig, int height_orig,
                  int padding_w, int padding_h) {
    if (top_x < padding_w) {
        return true;
    }
    else if (top_y < padding_h) {
        return true;
    }
    else if (top_y >= height_orig + padding_h) {
        return true;
    }
    else if (top_x >= width_orig + padding_w) {
        return true;
    }
    return false;
}

//this will transform input having multiple images stored contiguously
//Functionally incorrect now
void im2col_multiple_batches(const float *data_im, const int batch_size,
                             const int channels,
                             const int height, const int width, const int kernel_h, const int kernel_w,
                             const int pad_h, const int pad_w,
                             const int stride_h, const int stride_w,
                             float *data_col) {
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;
    int h, w,c, b;
    for (b = 0; b < batch_size; ++b) {
        for (c = 0; c < channels_col; ++c) {
            int w_offset = c % kernel_w;
            int h_offset = (c / kernel_w) % kernel_h;
            int c_im = c / kernel_h / kernel_w;
            for (h = 0; h < height_col; ++h) {
                for (w = 0; w < width_col; ++w) {
                    int h_pad = h * stride_h - pad_h + h_offset;
                    int w_pad = w * stride_w - pad_w + w_offset;
                    if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
                        data_col[(c * height_col + h) * width_col + w] =
                            data_im[(c_im * height + h_pad) * width + w_pad];
                        zendnnInfo(ZENDNN_ALGOLOG, "im2col_multiple_batches: ",
                                   (c * height_col + h) * width_col + w, " ",
                                   (c_im * height + h_pad) * width + w_pad);
                    }
                    else {
                        data_col[(c * height_col + h) * width_col + w] = 0;
                    }

                }
            }
        }
    }
    zendnnInfo(ZENDNN_ALGOLOG, "im2col_multiple_batches: ",
               (c * height_col + h) * width_col + w, " ",
               (c * height_col + h) * width_col + w, " ",
               height_col, " ", width_col, " ",
               channels_col);
}


//Caffe version of im2col...modified for few cases
void im2colNCHW(const float *data_im, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                float *data_col) {
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;
    int h, w,c;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int h_pad = h * stride_h - pad_h + h_offset;
                int w_pad = w * stride_w - pad_w + w_offset;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
                    data_col[(c * height_col + h) * width_col + w] =
                        data_im[(c_im * height + h_pad) * width + w_pad];
                    // printf("%d %d\n", (c * height_col + h) * width_col + w, (c_im * height + h_pad) * width + w_pad);
                }
                else {
                    data_col[(c * height_col + h) * width_col + w] = 0;
                }

            }
        }
    }
    //printf("%d %d %d %d %d\n\n", (c * height_col + h) * width_col + w, (c * height_col + h) * width_col + w, height_col, width_col, channels_col);

}



//Parallel version of im2col with OMP
void im2col_parNCHW(const float *data_im, const int channels,
                    const int height, const int width, const int kernel_h, const int kernel_w,
                    const int pad_h, const int pad_w,
                    const int stride_h, const int stride_w,
                    float *data_col) {
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;
    int c;
    #pragma omp parallel for private(c) schedule(dynamic)
    //printf("%d\n", omp_get_thread_num());
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int h_pad = h * stride_h - pad_h + h_offset;
                int w_pad = w * stride_w - pad_w + w_offset;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                    data_col[(c * height_col + h) * width_col + w] =
                        data_im[(c_im * height + h_pad) * width + w_pad];
                else {
                    data_col[(c * height_col + h) * width_col + w] = 0;
                }
            }
        }
    }
}


//based on Low-memory GEMM-based convolution algorithms for deep neural networks
//https://arxiv.org/pdf/1709.03395.pdf
//Defined in tensorflow
//https://github.com/tensorflow/tensorflow/blob/c501f6c1b479d28a538c10147c4f651ef68220c3/tensorflow/core/kernels/conv_grad_filter_ops.cc
void im2rowNHWC(const float *input_data, const int depth, const int height,
                const int width, const int filter_h, const int filter_w,
                const int pad_t, const int pad_l, const int pad_b, const int pad_r,
                const int stride_h, const int stride_w, float *col_data) {
    int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
    int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;

    int h = 0;
    int h_pad = -pad_t;
    for (h = 0; h < height_col; ++h) {
        int w_pad = -pad_l;
        for (int w = 0; w < width_col; ++w) {
            for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
                for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        int offset = (ih * width + iw) * depth;
                        #pragma omp simd
                        for (int k=0; k<depth; k++) {
                            col_data[k] = input_data[offset + k];
                        }
                    }
                    else {
                        // This should be simply padded with zero.
                        #pragma omp simd
                        for (int k=0; k<depth; k++) {
                            col_data[k] = 0;
                        }
                    }
                    col_data += depth;
                }
            }
            w_pad += stride_w;
        }
        h_pad += stride_h;
    }
}
void im2rowNHWCsplit(const float *input_data, const int depth, const int height,
                     const int width, const int filter_h, const int filter_w,
                     const int pad_t, const int pad_l, const int pad_b, const int pad_r,
                     const int stride_h, const int stride_w, float *col_data,
                     const int heightColOffset,
                     const int heightStart, const int no_of_threads) {

    int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;
    int out_width = ((width + pad_l + pad_r - filter_w) / stride_w + 1)*filter_h*
                    depth * filter_w;
    float *col_data_old = col_data;

    int h = 0;
    int h_pad = -pad_t;
    int w_pad = -pad_l;
    int h_offset = -pad_t;
    if (heightStart > 0) {
        h_offset = heightStart*stride_h-pad_t;
    }
    else {
        h_offset = -pad_t;
    }
    //To enable unrolling and better vectorization, (depth == 3) path unrolled the loop
    //along channel, this address first layer of convolution where no. of channels
    //in Image is 3.
    //For (depth%8 == 0), common case for other conv layers, vectorization is enabled by
    //to tell compiler to generate AVX256 SIMD instruction using (simd_blocks*8) loop.
    //Observed perf improvement with googlenet and alexnet.
    if (depth == 3) {
        #pragma omp parallel for num_threads(no_of_threads) private(h_pad, w_pad, col_data)
        for (int i = 0; i < heightColOffset; ++i) {
            w_pad = -pad_l;
            h_pad = h_offset + (i * stride_h);
            col_data = col_data_old + (i*out_width);
            for (int w = 0; w < width_col; ++w) {
                for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
                    for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            int offset = (ih * width + iw) * depth;
                            //#pragma omp simd
                            //for (int k=0; k<depth; k++) {
                            col_data[0] = input_data[offset + 0];
                            col_data[1] = input_data[offset + 1];
                            col_data[2] = input_data[offset + 2];
                            //}
                        }
                        else {
                            // This should be simply padded with zero.
                            //#pragma omp simd
                            //for (int k=0; k<depth; k++) {
                            col_data[0] = 0;
                            col_data[1] = 0;
                            col_data[2] = 0;
                            //}
                        }
                        col_data += depth;
                    }
                }
                w_pad += stride_w;
            }
        }
    }
    else if ((depth%8) == 0) {
        int simd_blocks = depth/8;
        #pragma omp parallel for num_threads(no_of_threads) private(h_pad, w_pad, col_data)
        for (int i = 0; i < heightColOffset; ++i) {
            w_pad = -pad_l;
            h_pad = h_offset + (i * stride_h);
            col_data = col_data_old + (i*out_width);
            for (int w = 0; w < width_col; ++w) {
                for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
                    for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            int offset = (ih * width + iw) * depth;
                            #pragma omp simd
                            for (int k=0; k<simd_blocks*8; k++) {
                                col_data[k] = input_data[offset + k];
                            }
                        }
                        else {
                            // This should be simply padded with zero.
                            #pragma omp simd
                            for (int k=0; k<simd_blocks*8; k++) {
                                col_data[k] = 0;
                            }
                        }
                        col_data += depth;
                    }
                }
                w_pad += stride_w;
            }
        }
    }
    else {
        #pragma omp parallel for num_threads(no_of_threads) private(h_pad, w_pad, col_data)
        for (int i = 0; i < heightColOffset; ++i) {
            w_pad = -pad_l;
            h_pad = h_offset + (i * stride_h);
            col_data = col_data_old + (i*out_width);
            for (int w = 0; w < width_col; ++w) {
                for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
                    for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            int offset = (ih * width + iw) * depth;
                            #pragma omp simd
                            for (int k=0; k<depth; k++) {
                                col_data[k] = input_data[offset + k];
                            }
                        }
                        else {
                            // This should be simply padded with zero.
                            #pragma omp simd
                            for (int k=0; k<depth; k++) {
                                col_data[k] = 0;
                            }
                        }
                        col_data += depth;
                    }
                }
                w_pad += stride_w;
            }
        }
    }
}

//based on Low-memory GEMM-based convolution algorithms for deep neural networks
//https://arxiv.org/pdf/1709.03395.pdf
//Defined in tensorflow amd modified for parallel version
//https://github.com/tensorflow/tensorflow/blob/c501f6c1b479d28a538c10147c4f651ef68220c3/tensorflow/core/kernels/conv_grad_filter_ops.cc
void im2rowNHWC_par(const float *input_data, const int depth, const int height,
                    const int width, const int filter_h, const int filter_w,
                    const int pad_t, const int pad_l, const int pad_b, const int pad_r,
                    const int stride_h, const int stride_w, float *col_data) {
    int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
    int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;

    int out_width = ((width + pad_l + pad_r - filter_w) / stride_w + 1)*filter_h*
                    depth * filter_w;
    float *col_data_old = col_data;


    int h = 0;
    int h_pad = -pad_t;
    int count = 0;
    #pragma omp parallel for private(h, col_data, h_pad)
    for (h = 0; h < height_col; ++h) {
        int w_pad = -pad_l;
        h_pad = -pad_t + (h * stride_h);
        col_data = col_data_old + (h*out_width);
        //int w_pad = -pad_l + (h * width_col * stride_w);
        //printf("Start %d\n", w_pad);
        for (int w = 0; w < width_col; ++w) {
            for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
                for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        memcpy(col_data, input_data + (ih * width + iw) * depth,
                               sizeof(float) * depth);
                    }
                    else {
                        // This should be simply padded with zero.
                        memset(col_data, 0, sizeof(float) * depth);
                        //printf("%d\n", depth);
                    }
                    col_data += depth;
                    count += depth;
                }
            }
            w_pad += stride_w;
        }
        //printf("End %d\n", w_pad);
    }
}

float timedifference_msec(struct timeval t0, struct timeval t1) {
    return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}


//OOP transpose for NHWC format when kernel is transposed
float *transpose(const float *matrix, int n, int m) {
    int i = 0;
    int j = 0;
    float num;
    float *transposed=(float *)malloc(sizeof(float)*n*m);
    if (transposed == NULL) {
        zendnnError(ZENDNN_ALGOLOG, "transpose Memory Error");
    }
    while (i < n) {
        j = 0;
        while (j < m) {
            num = *(matrix + i*m + j);
            *(transposed + i+n*j) = num;
            j++;
        }
        i++;
    }

    return transposed;
}


void *malloc_safe(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        return malloc_safe(size);
    }
    else {
        return ptr;
    }
}

void NCHW2NHWC(const float *nchw_data, int N, int C, int H, int W,
               float *nhwc_data) {
    for (int n = 0; n < N; n++) {
        int in_batch_offset = n * C * H * W;
        int out_batch_offset = n * H * W * C;
        for (int c = 0; c < C; ++c) {
            int in_ch_offset = c * H * W + in_batch_offset;
            int out_ch_offset = out_batch_offset + c;
            for (int h = 0; h < H; ++h) {
                int in_row_offset = h * W + in_ch_offset;
                int out_row_offset = out_ch_offset + h * W * C;
                for (int w = 0; w < W; ++w) {
                    int in_addr = w + in_row_offset;
                    int out_addr = out_row_offset + w * C;
                    nhwc_data[out_addr] = nchw_data[in_addr];
                }
            }
        }
    }
}

void NHWC2NCHW(const float *nhwc_data, int N, int C, int H, int W,

               float *nchw_data) {

    for (int n = 0; n < N; ++n) {

        int in_batch_offset = n * H * W * C;

        int out_batch_offset = n * C * H * W;

        for (int h = 0; h < H; ++h) {

            int in_row_offset = in_batch_offset + h * W * C;

            int out_row_offset = out_batch_offset + h * W;

            for (int w = 0; w < W; ++w) {

                int in_col_offset = in_row_offset + w * C;

                int out_col_offset = out_row_offset + w;

                for (int c = 0; c < C; ++c) {

                    int in_addr = in_col_offset + c;

                    int out_addr = out_col_offset + c * H * W;

                    nchw_data[out_addr] = nhwc_data[in_addr];
                }
            }
        }
    }
}

//Unrool im2row for kernel_width=3 and input_channel=3
void im2row_unrool_3x3(float *data_col_tmp, unsigned long data_col_offset,
                       const float *in_layer, unsigned long offset) {

    data_col_tmp[data_col_offset + 0] = in_layer[offset + 0];
    data_col_tmp[data_col_offset + 1] = in_layer[offset + 1];
    data_col_tmp[data_col_offset + 2] = in_layer[offset + 2];
    data_col_tmp[data_col_offset + 3] = in_layer[offset + 3];
    data_col_tmp[data_col_offset + 4] = in_layer[offset + 4];
    data_col_tmp[data_col_offset + 5] = in_layer[offset + 5];
    data_col_tmp[data_col_offset + 6] = in_layer[offset + 6];
    data_col_tmp[data_col_offset + 7] = in_layer[offset + 7];
    data_col_tmp[data_col_offset + 8] = in_layer[offset + 8];
}

//Unrool im2row for kernel_width=7 and input_channel=3
void im2row_unrool_7x3(float *data_col_tmp, unsigned long data_col_offset,
                       const float *in_layer, unsigned long offset) {

    data_col_tmp[data_col_offset + 0] = in_layer[offset + 0];
    data_col_tmp[data_col_offset + 1] = in_layer[offset + 1];
    data_col_tmp[data_col_offset + 2] = in_layer[offset + 2];
    data_col_tmp[data_col_offset + 3] = in_layer[offset + 3];
    data_col_tmp[data_col_offset + 4] = in_layer[offset + 4];
    data_col_tmp[data_col_offset + 5] = in_layer[offset + 5];
    data_col_tmp[data_col_offset + 6] = in_layer[offset + 6];
    data_col_tmp[data_col_offset + 7] = in_layer[offset + 7];
    data_col_tmp[data_col_offset + 8] = in_layer[offset + 8];
    data_col_tmp[data_col_offset + 9] = in_layer[offset + 9];
    data_col_tmp[data_col_offset + 10] = in_layer[offset + 10];
    data_col_tmp[data_col_offset + 11] = in_layer[offset + 11];
    data_col_tmp[data_col_offset + 12] = in_layer[offset + 12];
    data_col_tmp[data_col_offset + 13] = in_layer[offset + 13];
    data_col_tmp[data_col_offset + 14] = in_layer[offset + 14];
    data_col_tmp[data_col_offset + 15] = in_layer[offset + 15];
    data_col_tmp[data_col_offset + 16] = in_layer[offset + 16];
    data_col_tmp[data_col_offset + 17] = in_layer[offset + 17];
    data_col_tmp[data_col_offset + 18] = in_layer[offset + 18];
    data_col_tmp[data_col_offset + 19] = in_layer[offset + 19];
    data_col_tmp[data_col_offset + 20] = in_layer[offset + 20];
}

