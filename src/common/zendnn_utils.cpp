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

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <stdbool.h> // for padding_zone()
#include "common/zendnn_private.hpp"
#include "common/bfloat16.hpp"
#include "zendnn_logging.hpp"
#include "zendnn_helper.hpp"

using namespace zendnn;

void float_to_bf16(float *float_value, bfloat16 *bf16_val) {
    /*Set offset 2 to copy most significant 2 bytes of float
    to convert float values to bf16 values*/
    memcpy((bf16_val), (char *)(float_value) + 2, sizeof(int16_t));
}

float bf16_to_float(int16_t bf16_val) {
    int32_t inter_temp = *((int16_t *) &bf16_val);
    inter_temp = inter_temp << 16;
    float float_value = 0.0;
    memcpy(&float_value, &inter_temp, sizeof(int32_t));
    return float_value;
}

int cvt_int4_to_bf16(const int8_t *weights, int16_t *wei_bf16, int k, int n,
                     float *scales, int scale_size, int group_size,
                     zendnn_data_type_t scale_dt) {
    float *wei_f32 = (float *)zendnn_aligned_alloc(64, k*n*sizeof(float));
    #pragma omp parallel for
    for (int j=0; j<((k*n)/2) + 1;j++) {
        int idx_buff = 0;
        int weight_idx = j * 2;
        int val_idx = weight_idx / 2;
        int t1 = impl::int4_t::extract(weights[val_idx],
                                       impl::int4_extract_t::low_half);
        int t2 = impl::int4_t::extract(weights[val_idx],
                                       impl::int4_extract_t::high_half);
        if (weight_idx < k*n) {
            if (scale_size == 1) {
                wei_f32[weight_idx] = scales[0] * ((float)(t1));
            }
            else {
                idx_buff = (weight_idx / (group_size * n)) * n;
                int scale_offset = ((weight_idx%scale_size) % n) + idx_buff;
                wei_f32[weight_idx] = zendnn::impl::cpu::io::load_float_value(scale_dt, (void *)scales,
                            scale_offset) * ((float)(t1));
            }
        }
        weight_idx++;
        if (weight_idx < k*n) {
            if (scale_size == 1) {
                wei_f32[weight_idx] = scales[0] * ((float)(t2));
            }
            else {
                idx_buff = (weight_idx / (group_size * n)) * n;
                int scale_offset = ((weight_idx%scale_size) % n) + idx_buff;
                wei_f32[weight_idx] = zendnn::impl::cpu::io::load_float_value(scale_dt, (void *)scales,
                             scale_offset) * ((float)(t2));
            }
        }
    }
    cvt_float_to_bfloat16((impl::bfloat16_t *)wei_bf16, wei_f32, k*n);
    free(wei_f32);
    return 0;
}

int cvt_int8_to_bf16(const int8_t *weights, int16_t *wei_bf16, int k, int n,
                     float *scales, int scale_size, int group_size,
                     zendnn_data_type_t scale_dt) {
    float *wei_f32 = (float *)zendnn_aligned_alloc(64,k*n*sizeof(float));
    #pragma omp parallel for
    for (int i=0; i<k*n; i++) {
        if (scale_size == 1) {
            wei_f32[i] = scales[0] * (weights[i]);
        }
        else {
            int idx_buff = 0;
            idx_buff = (i / (group_size * n)) * n;
            int scale_offset = ((i%scale_size) % n) + idx_buff;
            wei_f32[i] = zendnn::impl::cpu::io::load_float_value(scale_dt, (void *)scales,
                         scale_offset) * (weights[i]);
        }
    }
    cvt_float_to_bfloat16((impl::bfloat16_t *)wei_bf16, wei_f32, k*n);
    free(wei_f32);
    return 0;
}

int cvt_int4_to_f32(const int8_t *weights, float *wei_f32, int k, int n,
                    float *scales, int scale_size, int group_size,
                    zendnn_data_type_t scale_dt) {
    #pragma omp parallel for
    for (int j=0; j<((k*n)/2) + 1; j++) {
        int idx_buff = 0;
        int weight_idx = j * 2;
        int val_idx = weight_idx / 2;
        int t1 = impl::int4_t::extract(weights[val_idx],
                                       impl::int4_extract_t::low_half);
        int t2 = impl::int4_t::extract(weights[val_idx],
                                       impl::int4_extract_t::high_half);
        if (weight_idx < k*n) {
            if (scale_size == 1) {
                wei_f32[weight_idx] = scales[0] * ((float)(t1));
            }
            else {
                idx_buff = (weight_idx / (group_size * n)) * n;
                int scale_offset = ((weight_idx%scale_size) % n) + idx_buff;
                wei_f32[weight_idx] = zendnn::impl::cpu::io::load_float_value(scale_dt, (void *)scales,
                            scale_offset) * ((float)(t1));
            }
        }
        weight_idx++;
        if (weight_idx < k*n) {
            if (scale_size == 1) {
                wei_f32[weight_idx] = scales[0] * ((float)(t2));
            }
            else {
                idx_buff = (weight_idx / (group_size * n)) * n;
                int scale_offset = ((weight_idx%scale_size) % n) + idx_buff;
                wei_f32[weight_idx] = zendnn::impl::cpu::io::load_float_value(scale_dt, (void *)scales,
                             scale_offset) * ((float)(t2));
            }
        }
    }
    return 0;
}

int cvt_int8_to_f32(const int8_t *weights, float *wei_f32, int k, int n,
                    float *scales, int scale_size, int group_size,
                    zendnn_data_type_t scale_dt) {
    #pragma omp parallel for
    for (int i=0; i<k*n; i++) {
        if (scale_size == 1) {
            wei_f32[i] = scales[0] * (weights[i]);
        }
        else {
            int idx_buff = 0;
            idx_buff = (i / (group_size * n)) * n;
            int scale_offset = ((i%scale_size) % n) + idx_buff;
            wei_f32[i] = zendnn::impl::cpu::io::load_float_value(scale_dt, (void *)scales,
                         scale_offset) * (weights[i]);
        }
    }
    return 0;
}

// initialize memory pool static array for use by the kernels
// declared in zendnn_utils.hpp
ZenLibMemoryPool *ZenLibMemoryPool::zenLibMemPoolArr[ZEN_LIB_MEM_POOL_LIMIT] = {NULL};
int ZenLibMemoryPool::zenLibMemPoolCount = 0;


//ZenDNN Env Instance
zendnnEnv readEnv() {
    const zendnnEnv &obj = zendnnEnv::ZenDNNEnv();
    return (obj);
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

// Used in zenConvolution2Dbase_LPGEMM1x1() (u8, s8, s32)
// TODO: Verify correctness of this function; currently, not in use.
void im2rowNHWCsplit_lpgemm(const uint8_t *input_data, const int depth,
                            const int height,
                            const int width, const int filter_h, const int filter_w,
                            const int pad_t, const int pad_l, const int pad_b, const int pad_r,
                            const int stride_h, const int stride_w, uint8_t *col_data,
                            const int heightColOffset,
                            const int heightStart, const int no_of_threads) {

    int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;
    int out_width = ((width + pad_l + pad_r - filter_w) / stride_w + 1)*filter_h*
                    depth * filter_w;
    uint8_t *col_data_old = col_data;

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

