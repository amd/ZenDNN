/*******************************************************************************
* Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <cmath>
#include <cassert>
#include <cblas.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include "zendnn_convolution_winograd.hpp"
#include "zendnn_logging.hpp"
#include <omp.h>

using namespace zendnn;

void filter_transform_2x2_3x3(zendnnEnv zenEnvObj, const float *filter,
                              const int num_channels, const int num_filters, float *out) {

    //this assumes that the filter is in HWCN format.
    //If the filter plane is g (3x3), then the operation that needs to be performed is GgGT
    //where G = [ 1, 0, 0
    //            0.5, 0.5, 0.5,
    //            0.5, -0.5, 0.5,
    //            0, 0, 1]
    //
    //      GT = [ 1, 0.5, 0.5, 0,
    //             0, 0.5, -0.5, 0.5,
    //             0, 0.5, 0.5, 1]
    //Output is 4x4
    //Final output should be Nx4x4xC

    const int C = num_channels;
    const int K = num_filters;
    const int FW = 3; //filter width
    const int OW = 4; //output width
    const int limc = C - (C % 8);
    int k;

    #pragma omp parallel for
    for (k = 0; k < num_filters; k++) {
        //ROME has 256 bit width registers.
        //Thus, we group num_channels into groups of 8 (256 / 32)
        float *V = out + k * 4 * 4 * num_channels;
        float Gg[4][3][8];
        int c, ci, i, j;

        //simplify the operations
        for (c = 0; c < limc; c+=8) {
            for (j = 0, ci = c; j < 8; j++, ci++) {
                Gg[0][0][j] = AT_HWCN(filter, FW, C, K, 0, 0, ci, k);
                Gg[0][1][j] = AT_HWCN(filter, FW, C, K, 0, 1, ci, k);
                Gg[0][2][j] = AT_HWCN(filter, FW, C, K, 0, 2, ci, k);

                Gg[1][0][j] = (AT_HWCN(filter, FW, C, K, 0, 0, ci, k) + AT_HWCN(filter, FW, C,
                               K, 1, 0, ci, k) + AT_HWCN(filter, FW, C, K, 2, 0, ci, k))*0.5f;
                Gg[1][1][j] = (AT_HWCN(filter, FW, C, K, 0, 1, ci, k) + AT_HWCN(filter, FW, C,
                               K, 1, 1, ci, k) + AT_HWCN(filter, FW, C, K, 2, 1, ci, k))*0.5f;
                Gg[1][2][j] = (AT_HWCN(filter, FW, C, K, 0, 2, ci, k) + AT_HWCN(filter, FW, C,
                               K, 1, 2, ci, k) + AT_HWCN(filter, FW, C, K, 2, 2, ci, k))*0.5f;

                Gg[2][0][j] = Gg[1][0][j] - AT_HWCN(filter, FW, C, K, 1, 0, ci, k);
                Gg[2][1][j] = Gg[1][1][j] - AT_HWCN(filter, FW, C, K, 1, 1, ci, k);
                Gg[2][2][j] = Gg[1][2][j] - AT_HWCN(filter, FW, C, K, 1, 2, ci, k);

                Gg[3][0][j] = AT_HWCN(filter, FW, C, K, 2, 0, ci, k);
                Gg[3][1][j] = AT_HWCN(filter, FW, C, K, 2, 1, ci, k);
                Gg[3][2][j] = AT_HWCN(filter, FW, C, K, 2, 2, ci, k);
            }

            for (j = 0, ci = c; j < 8; j++, ci++) {
                AT(V,C,OW,0,0,ci) = Gg[0][0][j];
                AT(V,C,OW,0,1,ci) = (Gg[0][0][j] + Gg[0][1][j] + Gg[0][2][j]) * 0.5f;
                AT(V,C,OW,0,2,ci) = AT(V,C,OW,0,1,ci) - Gg[0][1][j];
                AT(V,C,OW,0,3,ci) = Gg[0][2][j];

                AT(V,C,OW,1,0,ci) = Gg[1][0][j];
                AT(V,C,OW,1,1,ci) = (Gg[1][0][j] + Gg[1][1][j] + Gg[1][2][j]) * 0.5f;
                AT(V,C,OW,1,2,ci) = AT(V,C,OW,1,1,ci) - Gg[1][1][j];
                AT(V,C,OW,1,3,ci) = Gg[1][2][j];

                AT(V,C,OW,2,0,ci) = Gg[2][0][j];
                AT(V,C,OW,2,1,ci) = (Gg[2][0][j] + Gg[2][1][j] + Gg[2][2][j]) * 0.5f;
                AT(V,C,OW,2,2,ci) = AT(V,C,OW,2,1,ci) - Gg[2][1][j];
                AT(V,C,OW,2,3,ci) = Gg[2][2][j];

                AT(V,C,OW,3,0,ci) = Gg[3][0][j];
                AT(V,C,OW,3,1,ci) = (Gg[3][0][j] + Gg[3][1][j] + Gg[3][2][j]) * 0.5f;
                AT(V,C,OW,3,2,ci) = AT(V,C,OW,3,1,ci) - Gg[3][1][j];
                AT(V,C,OW,3,3,ci) = Gg[3][2][j];
            }
        }

        // handle the remaining num_channels in a non-vectorized way for now
        for (j=0, ci = limc; ci < num_channels; ci++, j++) {
            Gg[0][0][j] = AT_HWCN(filter, FW, C, K, 0, 0, ci, k);
            Gg[0][1][j] = AT_HWCN(filter, FW, C, K, 0, 1, ci, k);
            Gg[0][2][j] = AT_HWCN(filter, FW, C, K, 0, 2, ci, k);

            Gg[1][0][j] = (AT_HWCN(filter, FW, C, K, 0, 0, ci, k) + AT_HWCN(filter, FW, C,
                           K, 1, 0, ci, k) + AT_HWCN(filter, FW, C, K, 2, 0, ci, k))*0.5f;
            Gg[1][1][j] = (AT_HWCN(filter, FW, C, K, 0, 1, ci, k) + AT_HWCN(filter, FW, C,
                           K, 1, 1, ci, k) + AT_HWCN(filter, FW, C, K, 2, 1, ci, k))*0.5f;
            Gg[1][2][j] = (AT_HWCN(filter, FW, C, K, 0, 2, ci, k) + AT_HWCN(filter, FW, C,
                           K, 1, 2, ci, k) + AT_HWCN(filter, FW, C, K, 2, 2, ci, k))*0.5f;

            Gg[2][0][j] = Gg[1][0][j] - AT_HWCN(filter, FW, C, K, 1, 0, ci, k);
            Gg[2][1][j] = Gg[1][1][j] - AT_HWCN(filter, FW, C, K, 1, 1, ci, k);
            Gg[2][2][j] = Gg[1][2][j] - AT_HWCN(filter, FW, C, K, 1, 2, ci, k);

            Gg[3][0][j] = AT_HWCN(filter, FW, C, K, 2, 0, ci, k);
            Gg[3][1][j] = AT_HWCN(filter, FW, C, K, 2, 1, ci, k);
            Gg[3][2][j] = AT_HWCN(filter, FW, C, K, 2, 2, ci, k);
        }

        for (j=0, ci = limc; ci < num_channels; ci++, j++) {
            AT(V,C,OW,0,0,ci) = Gg[0][0][j];
            AT(V,C,OW,0,1,ci) = (Gg[0][0][j] + Gg[0][1][j] + Gg[0][2][j]) * 0.5f;
            AT(V,C,OW,0,2,ci) = AT(V,C,OW,0,1,ci) - Gg[0][1][j];
            AT(V,C,OW,0,3,ci) = Gg[0][2][j];

            AT(V,C,OW,1,0,ci) = Gg[1][0][j];
            AT(V,C,OW,1,1,ci) = (Gg[1][0][j] + Gg[1][1][j] + Gg[1][2][j]) * 0.5f;
            AT(V,C,OW,1,2,ci) = AT(V,C,OW,1,1,ci) - Gg[1][1][j];
            AT(V,C,OW,1,3,ci) = Gg[1][2][j];

            AT(V,C,OW,2,0,ci) = Gg[2][0][j];
            AT(V,C,OW,2,1,ci) = (Gg[2][0][j] + Gg[2][1][j] + Gg[2][2][j]) * 0.5f;
            AT(V,C,OW,2,2,ci) = AT(V,C,OW,2,1,ci) - Gg[2][1][j];
            AT(V,C,OW,2,3,ci) = Gg[2][2][j];

            AT(V,C,OW,3,0,ci) = Gg[3][0][j];
            AT(V,C,OW,3,1,ci) = (Gg[3][0][j] + Gg[3][1][j] + Gg[3][2][j]) * 0.5f;
            AT(V,C,OW,3,2,ci) = AT(V,C,OW,3,1,ci) - Gg[3][1][j];
            AT(V,C,OW,3,3,ci) = Gg[3][2][j];
        }
    }

}

void input_transform_2x2_3x3(zendnnEnv zenEnvObj, const float *input,
                             const int batch_size,
                             const int height, const int width, const int num_channels,
                             const int pad_t, const int pad_l, const int pad_b, const int pad_r,
                             float *out, const int num_tiles, const int output_height,
                             const int output_width) {

    //This assumes that the input is in NHWC format, and is not tiled.
    //The input tile size will be 4x4. Operation to be performed is BTdB
    //where BT = [1, 0, -1, 0,
    //            0, 1, 1, 0
    //            0, -1, 1, 0
    //            0, 1, 0, -1]
    //      B = [1, 0, 0, 0,
    //           0, 1, -1, 1,
    //           -1, 1, 1, 0,
    //            0, 0, 0, -1]

    const int C = num_channels;
    const int TW = 4;
    const int limc = C - (C % 8);
    int n,h,w;
    const int num_tiles_per_image = num_tiles / batch_size;
    const int num_tiles_per_row = std::ceil(output_width / 2);

    #pragma omp parallel for collapse(3) private(n,h,w)
    for (n = 0; n < batch_size; n++) {
        // one image at a time
        //num_tiles_generated_by_one_image = H/2 * W /2
        //and tiles overlap by 2, with its size 4x4
        //equivalent to stride 2
        for (h = -pad_t; h < (height + pad_b)-3; h+=2) {
            for (w = -pad_l; w < (width + pad_r)-3; w+=2) {
                float x[4][4][num_channels];
                float BTx[4][4][8];
                int th = h + pad_t;
                int tw = w + pad_l;
                unsigned long t = (unsigned long)n * num_tiles_per_image +
                                  (th / 2) * num_tiles_per_row +
                                  (tw / 2); //tile_counter
                const float *TI = input + (unsigned long)n * height * width * num_channels;
                float *U = out + t * 4 * 4 * num_channels;
                int i,j,c,ci;
                int hi, wj;

                int start_h, end_h, start_w, end_w;
                start_h = h < 0 ? -h : 0; // assuming pad does not exceed 4
                end_h = h + 4 > height ? height - h : 4;
                start_w = w < 0 ? -w : 0;
                end_w = w + 4 > width ? width - w: 4;

                // copy from input only in valid locations
                for (i = start_h, hi = h + i; i < end_h; i++, hi++) {
                    for (j = start_w, wj = w + j; j < end_w; j++, wj++) {
                        for (c = 0; c < num_channels; c++) {
                            x[i][j][c] = AT(TI, num_channels, width, hi, wj, c);
                        }
                    }
                }

                //pad the remaining regions
                // for large H,W, these loops will not be entered most of the time
                for (i = 0; i < start_h; i++) for (j = 0; j < 4; j++) for (c= 0;
                                c < num_channels; c++) {
                            x[i][j][c] = 0;
                        }
                for (i = end_h; i < 4; i++) for (j = 0; j < 4; j++) for (c= 0; c < num_channels;
                                c++) {
                            x[i][j][c] = 0;
                        }
                for (j = 0; j < start_w; j++) for (i = 0; i < 4; i++) for (c= 0;
                                c < num_channels; c++) {
                            x[i][j][c] = 0;
                        }
                for (j = end_w; j < 4; j++) for (i = 0; i < 4; i++) for (c= 0; c < num_channels;
                                c++) {
                            x[i][j][c] = 0;
                        }

                //  simplify the operations
                for (c = 0; c < limc; c += 8) {
                    for (j = 0, ci = c; j < 8; j++, ci++) {
                        BTx[0][0][j] = x[0][0][ci] - x[2][0][ci];
                        BTx[0][1][j] = x[0][1][ci] - x[2][1][ci];
                        BTx[0][2][j] = x[0][2][ci] - x[2][2][ci];
                        BTx[0][3][j] = x[0][3][ci] - x[2][3][ci];

                        BTx[1][0][j] = x[1][0][ci] + x[2][0][ci];
                        BTx[1][1][j] = x[1][1][ci] + x[2][1][ci];
                        BTx[1][2][j] = x[1][2][ci] + x[2][2][ci];
                        BTx[1][3][j] = x[1][3][ci] + x[2][3][ci];

                        BTx[2][0][j] = -x[1][0][ci] + x[2][0][ci];
                        BTx[2][1][j] = -x[1][1][ci] + x[2][1][ci];
                        BTx[2][2][j] = -x[1][2][ci] + x[2][2][ci];
                        BTx[2][3][j] = -x[1][3][ci] + x[2][3][ci];

                        BTx[3][0][j] = x[1][0][ci] - x[3][0][ci];
                        BTx[3][1][j] = x[1][1][ci] - x[3][1][ci];
                        BTx[3][2][j] = x[1][2][ci] - x[3][2][ci];
                        BTx[3][3][j] = x[1][3][ci] - x[3][3][ci];
                    }

                    for (j = 0, ci = c; j < 8; j++, ci++) {
                        AT(U, C, TW, 0, 0, ci) = BTx[0][0][j] - BTx[0][2][j];
                        AT(U, C, TW, 0, 1, ci) = BTx[0][1][j] + BTx[0][2][j];
                        AT(U, C, TW, 0, 2, ci) = -BTx[0][1][j] + BTx[0][2][j];
                        AT(U, C, TW, 0, 3, ci) = BTx[0][1][j] - BTx[0][3][j];

                        AT(U, C, TW, 1, 0, ci) = BTx[1][0][j] - BTx[1][2][j];
                        AT(U, C, TW, 1, 1, ci) = BTx[1][1][j] + BTx[1][2][j];
                        AT(U, C, TW, 1, 2, ci) = -BTx[1][1][j] + BTx[1][2][j];
                        AT(U, C, TW, 1, 3, ci) = BTx[1][1][j] - BTx[1][3][j];

                        AT(U, C, TW, 2, 0, ci) = BTx[2][0][j] - BTx[2][2][j];
                        AT(U, C, TW, 2, 1, ci) = BTx[2][1][j] + BTx[2][2][j];
                        AT(U, C, TW, 2, 2, ci) = -BTx[2][1][j] + BTx[2][2][j];
                        AT(U, C, TW, 2, 3, ci) = BTx[2][1][j] - BTx[2][3][j];

                        AT(U, C, TW, 3, 0, ci) = BTx[3][0][j] - BTx[3][2][j];
                        AT(U, C, TW, 3, 1, ci) = BTx[3][1][j] + BTx[3][2][j];
                        AT(U, C, TW, 3, 2, ci) = -BTx[3][1][j] + BTx[3][2][j];
                        AT(U, C, TW, 3, 3, ci) = BTx[3][1][j] - BTx[3][3][j];
                    }
                }

                // handle the remaining num_channels in a non-vectorized way
                for (j = 0, ci = limc; ci < num_channels; ci++, j++) {
                    BTx[0][0][j] = x[0][0][ci] - x[2][0][ci];
                    BTx[0][1][j] = x[0][1][ci] - x[2][1][ci];
                    BTx[0][2][j] = x[0][2][ci] - x[2][2][ci];
                    BTx[0][3][j] = x[0][3][ci] - x[2][3][ci];

                    BTx[1][0][j] = x[1][0][ci] + x[2][0][ci];
                    BTx[1][1][j] = x[1][1][ci] + x[2][1][ci];
                    BTx[1][2][j] = x[1][2][ci] + x[2][2][ci];
                    BTx[1][3][j] = x[1][3][ci] + x[2][3][ci];

                    BTx[2][0][j] = -x[1][0][ci] + x[2][0][ci];
                    BTx[2][1][j] = -x[1][1][ci] + x[2][1][ci];
                    BTx[2][2][j] = -x[1][2][ci] + x[2][2][ci];
                    BTx[2][3][j] = -x[1][3][ci] + x[2][3][ci];

                    BTx[3][0][j] = x[1][0][ci] - x[3][0][ci];
                    BTx[3][1][j] = x[1][1][ci] - x[3][1][ci];
                    BTx[3][2][j] = x[1][2][ci] - x[3][2][ci];
                    BTx[3][3][j] = x[1][3][ci] - x[3][3][ci];

                    AT(U, C, TW, 0, 0, ci) = BTx[0][0][j] - BTx[0][2][j];
                    AT(U, C, TW, 0, 1, ci) = BTx[0][1][j] + BTx[0][2][j];
                    AT(U, C, TW, 0, 2, ci) = -BTx[0][1][j] + BTx[0][2][j];
                    AT(U, C, TW, 0, 3, ci) = BTx[0][1][j] - BTx[0][3][j];

                    AT(U, C, TW, 1, 0, ci) = BTx[1][0][j] - BTx[1][2][j];
                    AT(U, C, TW, 1, 1, ci) = BTx[1][1][j] + BTx[1][2][j];
                    AT(U, C, TW, 1, 2, ci) = -BTx[1][1][j] + BTx[1][2][j];
                    AT(U, C, TW, 1, 3, ci) = BTx[1][1][j] - BTx[1][3][j];

                    AT(U, C, TW, 2, 0, ci) = BTx[2][0][j] - BTx[2][2][j];
                    AT(U, C, TW, 2, 1, ci) = BTx[2][1][j] + BTx[2][2][j];
                    AT(U, C, TW, 2, 2, ci) = -BTx[2][1][j] + BTx[2][2][j];
                    AT(U, C, TW, 2, 3, ci) = BTx[2][1][j] - BTx[2][3][j];

                    AT(U, C, TW, 3, 0, ci) = BTx[3][0][j] - BTx[3][2][j];
                    AT(U, C, TW, 3, 1, ci) = BTx[3][1][j] + BTx[3][2][j];
                    AT(U, C, TW, 3, 2, ci) = -BTx[3][1][j] + BTx[3][2][j];
                    AT(U, C, TW, 3, 3, ci) = BTx[3][1][j] - BTx[3][3][j];
                }
            }
        }
    }
}

void batched_gemm_2x2_3x3(zendnnEnv zenEnvObj, float *transformed_image,
                          const int num_tiles,
                          const int num_channels, const int num_images,
                          float *transformed_filter, const int num_filters, float *out) {

    /*
      The third step in winograd algorithm is an element-wise multiply accumulate
      As described in the paper, this can be cast as a gemm operation.

      Note that transformed_image and transformed_filter are both
      of the form NHWC with H=W=4.
      Let T be the number of tiles, K be the number of filters.
      Consider the first value in T1 across all channels - T1(0,0,c).
      Similarly K1(0,0,c).
      To fill the first value in the intermediate output (before out_transform),
      we must do: for(c = 0; c < num_channels; c++) O += T1(0,0,c) * K1(0,0,c) -
      this is required by the algorithm.

      Observe that this can be extended into a gemm product where the first matrix
      is T * C, with each row being Ti(0,0,0) -> Ti(0,0,num_channels). The second
      matrix is C * K with each column being Ki(0,0,0) -> Ki(0,0,num_channels). There
      will be 16 such matrix multiplications ( Ti(0,0,c) -> Ti(3,3,c) ).

      With some careful manipulation of the parameters for sgemm call, we can avoid
      having to do any explicit data transformation from N*4*4*C to those matrices.
    */

    int i, j;
    const float alpha = 1.0f;
    const float beta =  0.0f;
    const int m = num_tiles;
    const int k = num_channels;
    const int n = num_filters;
    const int lda = 16 * num_channels;
    const int ldb = 16 * num_channels;
    const int ldc = 16 * num_filters;

    unsigned int thread_qty = zenEnvObj.omp_num_threads;
#if BLIS_EXPERT
    //VggNet exposed gemm sizes which scales well till 6 threads
    //6 and 8 are divisible by #cores AMD CPUs generally have 8, 24, 32, 64
    //TBD: Need to run with different configuration to get ideal no. for blis_num_threads
    int blis_num_threads = thread_qty<6?thread_qty:6;
    if (thread_qty > blis_num_threads && (thread_qty%blis_num_threads != 0)) {
        blis_num_threads = 8;
    }
    thread_qty = (thread_qty%blis_num_threads)==0?(thread_qty/blis_num_threads):
                 (thread_qty/blis_num_threads)+1;
    omp_set_max_active_levels(2);
    #pragma omp parallel for num_threads(thread_qty)
#else
    omp_set_max_active_levels(1);
    #pragma omp parallel for
#endif
    for (i = 0; i < 16; i++) {
        float *image = transformed_image + i * num_channels;
        float *filter = transformed_filter + i * num_channels;
        float *output = out + i * num_filters;
#if BLIS_EXPERT
        if ((thread_qty%blis_num_threads)!=0 && omp_get_num_threads()==(thread_qty-1)) {
            blis_num_threads = thread_qty%blis_num_threads;
        }
        //creating blis expert interface
        blis_expert blis_obj(blis_num_threads, BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE);
        bli_obj_create_with_attached_buffer(blis_obj.dt, m, k, image, lda, 1,
                                            &blis_obj.a);
        bli_obj_create_with_attached_buffer(blis_obj.dt, k, n, filter, 1, ldb,
                                            &blis_obj.b);
        bli_obj_create_with_attached_buffer(blis_obj.dt, m, n, output, ldc, 1,
                                            &blis_obj.c);

        bli_gemm_ex(&blis_obj.alpha, &blis_obj.a, &blis_obj.b, &blis_obj.beta,
                    &blis_obj.c, NULL, &blis_obj.rntm);
#else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, n, k, alpha,
                    image, lda,
                    filter, ldb,
                    beta,
                    output, ldc
                   );
#endif
    }
}

void out_transform_2x2_3x3(zendnnEnv zenEnvObj, float *tiled_input,
                           const int num_tiles, const int num_channels,
                           float *out, const int batch_size, const int output_height,
                           const int output_width, bool sum_fused) {
    /*
    tiled_input is assumed to be the result of gemm call. It will be of the format Rx4x4xM, where
    R is the number of image tiles, and M is the number of filters

    This function performs the operation ATmA, where
      m is a 4x4 matrix from tiled_input at a particular tile and channel,
      AT = [1, 1, 1, 0,
            0, 1, -1, -1]
      A = [1, 0,
           1, 1,
           1, -1,
           0, -1]

      The output of ATmA will be 2x2

      This function also places this in the right order back into the original non-tiled format
      of NHWC of the image.
      Hence, output of function will be of the form NxHxWxnum_channels
    */

    int n,h,w;
    const int TW = 4; //tile width
    const int num_tiles_per_image = num_tiles / batch_size;
    const int limc = num_channels - (num_channels % 8);
    const int num_tiles_per_row = std::ceil(output_width / 2);

    #pragma omp parallel for collapse(3) private(n,h,w)
    for (n = 0; n < batch_size; n++) {
        for (h = 0; h < output_height; h+=2) {
            for (w = 0; w < output_width; w+=2) {
                float ATm[2][4][8];
                int c, ci, i, j;
                unsigned long tile_counter = (unsigned long)n * num_tiles_per_image +
                                             (h / 2) * num_tiles_per_row +
                                             (w / 2);
                const float *I = tiled_input + tile_counter * TW * TW * num_channels;
                float *O = out + (unsigned long)n * output_height * output_width * num_channels
                           + h *
                           output_width * num_channels + w * num_channels;

                // simplify the operations
                for (c = 0; c < limc; c+=8) {
                    for (j = 0, ci = c; j < 8; j++, ci++) {
                        //calculate AT * m
                        ATm[0][0][j] = AT(I, num_channels, TW, 0, 0, ci) + AT(I, num_channels, TW, 1, 0,
                                       ci) + AT(I, num_channels, TW, 2, 0, ci);
                        ATm[0][1][j] = AT(I, num_channels, TW, 0, 1, ci) + AT(I, num_channels, TW, 1, 1,
                                       ci) + AT(I, num_channels, TW, 2, 1, ci);
                        ATm[0][2][j] = AT(I, num_channels, TW, 0, 2, ci) + AT(I, num_channels, TW, 1, 2,
                                       ci) + AT(I, num_channels, TW, 2, 2, ci);
                        ATm[0][3][j] = AT(I, num_channels, TW, 0, 3, ci) + AT(I, num_channels, TW, 1, 3,
                                       ci) + AT(I, num_channels, TW, 2, 3, ci);

                        ATm[1][0][j] = AT(I, num_channels, TW, 1, 0, ci) - AT(I, num_channels, TW, 2, 0,
                                       ci) - AT(I, num_channels, TW, 3, 0, ci);
                        ATm[1][1][j] = AT(I, num_channels, TW, 1, 1, ci) - AT(I, num_channels, TW, 2, 1,
                                       ci) - AT(I, num_channels, TW, 3, 1, ci);
                        ATm[1][2][j] = AT(I, num_channels, TW, 1, 2, ci) - AT(I, num_channels, TW, 2, 2,
                                       ci) - AT(I, num_channels, TW, 3, 2, ci);
                        ATm[1][3][j] = AT(I, num_channels, TW, 1, 3, ci) - AT(I, num_channels, TW, 2, 3,
                                       ci) - AT(I, num_channels, TW, 3, 3, ci);
                    }

                    //calculate (AT * m) * A and scatter the tile to the output tensor
                    if (sum_fused) {
                        for (j = 0, ci = c; j < 8; j++, ci++) {
                            AT(O, num_channels, output_width, 0,0,
                               ci) += ATm[0][0][j] + ATm[0][1][j] + ATm[0][2][j];
                            AT(O, num_channels, output_width, 0,1,
                               ci) += ATm[0][1][j] - ATm[0][2][j] - ATm[0][3][j];

                            AT(O, num_channels, output_width, 1,0,
                               ci) += ATm[1][0][j] + ATm[1][1][j] + ATm[1][2][j];
                            AT(O, num_channels, output_width, 1,1,
                               ci) += ATm[1][1][j] - ATm[1][2][j] - ATm[1][3][j];
                        }
                    }
                    else {
                        for (j = 0, ci = c; j < 8; j++, ci++) {
                            AT(O, num_channels, output_width, 0,0,
                               ci) = ATm[0][0][j] + ATm[0][1][j] + ATm[0][2][j];
                            AT(O, num_channels, output_width, 0,1,
                               ci) = ATm[0][1][j] - ATm[0][2][j] - ATm[0][3][j];

                            AT(O, num_channels, output_width, 1,0,
                               ci) = ATm[1][0][j] + ATm[1][1][j] + ATm[1][2][j];
                            AT(O, num_channels, output_width, 1,1,
                               ci) = ATm[1][1][j] - ATm[1][2][j] - ATm[1][3][j];
                        }
                    }

                }
                //handle the remaining num_channels in a non-vectorized way
                for (j = 0, ci = limc; ci < num_channels; j++, ci++) {
                    //calculate AT * m
                    ATm[0][0][j] = AT(I, num_channels, TW, 0, 0, ci) + AT(I, num_channels, TW, 1, 0,
                                   ci) + AT(I, num_channels, TW, 2, 0, ci);
                    ATm[0][1][j] = AT(I, num_channels, TW, 0, 1, ci) + AT(I, num_channels, TW, 1, 1,
                                   ci) + AT(I, num_channels, TW, 2, 1, ci);
                    ATm[0][2][j] = AT(I, num_channels, TW, 0, 2, ci) + AT(I, num_channels, TW, 1, 2,
                                   ci) + AT(I, num_channels, TW, 2, 2, ci);
                    ATm[0][3][j] = AT(I, num_channels, TW, 0, 3, ci) + AT(I, num_channels, TW, 1, 3,
                                   ci) + AT(I, num_channels, TW, 2, 3, ci);

                    ATm[1][0][j] = AT(I, num_channels, TW, 1, 0, ci) - AT(I, num_channels, TW, 2, 0,
                                   ci) - AT(I, num_channels, TW, 3, 0, ci);
                    ATm[1][1][j] = AT(I, num_channels, TW, 1, 1, ci) - AT(I, num_channels, TW, 2, 1,
                                   ci) - AT(I, num_channels, TW, 3, 1, ci);
                    ATm[1][2][j] = AT(I, num_channels, TW, 1, 2, ci) - AT(I, num_channels, TW, 2, 2,
                                   ci) - AT(I, num_channels, TW, 3, 2, ci);
                    ATm[1][3][j] = AT(I, num_channels, TW, 1, 3, ci) - AT(I, num_channels, TW, 2, 3,
                                   ci) - AT(I, num_channels, TW, 3, 3, ci);

                    //calculate (AT * m) * A and scatter the tile to the output tensor
                    if (sum_fused) {
                        AT(O, num_channels, output_width, 0,0,
                           ci) += ATm[0][0][j] + ATm[0][1][j] + ATm[0][2][j];
                        AT(O, num_channels, output_width, 0,1,
                           ci) += ATm[0][1][j] - ATm[0][2][j] - ATm[0][3][j];

                        AT(O, num_channels, output_width, 1,0,
                           ci) += ATm[1][0][j] + ATm[1][1][j] + ATm[1][2][j];
                        AT(O, num_channels, output_width, 1,1,
                           ci) += ATm[1][1][j] - ATm[1][2][j] - ATm[1][3][j];
                    }
                    else {
                        AT(O, num_channels, output_width, 0,0,
                           ci) = ATm[0][0][j] + ATm[0][1][j] + ATm[0][2][j];
                        AT(O, num_channels, output_width, 0,1,
                           ci) = ATm[0][1][j] - ATm[0][2][j] - ATm[0][3][j];

                        AT(O, num_channels, output_width, 1,0,
                           ci) = ATm[1][0][j] + ATm[1][1][j] + ATm[1][2][j];
                        AT(O, num_channels, output_width, 1,1,
                           ci) = ATm[1][1][j] - ATm[1][2][j] - ATm[1][3][j];
                    }
                }
            }
        }
    }

}

void post_conv_transform(const int batch_size, const int output_height,
                         const int output_width, const int num_channels,
                         float *out,
                         const float *bias, const bool relu, const float *scale) {

    int m;
    const unsigned long  total_size = (unsigned long)batch_size * output_height *
                                      output_width * num_channels;

    // move if conditions outside for better performance
    if (bias != NULL && relu == true && scale != NULL) {
        #pragma omp parallel for
        for (m = 0; m < total_size; m += num_channels)
            for (int c = 0; c < num_channels; c++) {
                out[ m + c ] = out[ m + c] * scale[c] + bias[c];
                out[ m + c ] = out[ m + c] > 0 ? out[ m + c] : 0;
            }
    }
    else if (bias != NULL && relu == false && scale != NULL) {
        #pragma omp parallel for
        for (m = 0; m < total_size; m += num_channels)
            for (int c = 0; c < num_channels; c++) {
                out[ m + c ] = out[ m + c] * scale[c] + bias[c];
            }
    }
    else if (bias != NULL && relu == true && scale == NULL) {
        #pragma omp parallel for
        for (m = 0; m < total_size; m += num_channels)
            for (int c = 0; c < num_channels; c++) {
                out[ m + c ] = out[ m + c] + bias[c];
                out[ m + c ] = out[ m + c] > 0 ? out[ m + c] : 0;
            }
    }
    else if (bias != NULL && relu == false && scale == NULL) {
        #pragma omp parallel for
        for (m = 0; m < total_size; m += num_channels)
            for (int c = 0; c < num_channels; c++) {
                out[ m + c ] = out[ m + c] + bias[c];
            }
    }
    else if (bias == NULL && relu == true && scale == NULL) {
        #pragma omp parallel for
        for (m = 0; m < total_size; m += num_channels)
            for (int c = 0; c < num_channels; c++) {
                out[ m + c ] = out[ m + c] > 0 ? out[ m + c] : 0;
            }
    }
}

void winograd_2x2_3x3(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int num_images,
    const int num_channels,
    const int height,
    const int width,
    const float *filter,
    const int num_filters,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const float *bias,
    float *out_layer,
    const int out_height,
    const int out_width,
    const bool relu,
    const bool sum_fused,
    const float *scale
) {
    assert((kernel_h == 3) && (kernel_w == 3) &&
           "Winograd kernel called for non 3x3 filter");

    // number of tiles
    const int P = num_images * std::ceil(out_height * 0.5) * std::ceil(
                      out_width * 0.5);

    static unsigned long MAX_IMAGE_TILES = (unsigned long)(P+1) * 4 * 4 *
                                           num_channels; // initialization happens only once
    static unsigned long MAX_FILTER_TILES = (unsigned long)(
            num_filters+1) * 4 * 4 * num_channels;
    static unsigned long MAX_OUTPUT_TILES = (unsigned long)(
            P+1) * 4 * 4 * num_filters;

    unsigned long current_image_tiles = (unsigned long)(P+1) * 4 * 4 * num_channels;
    unsigned long current_filter_tiles = (unsigned long)(num_filters+1) * 4 * 4 *
                                         num_channels;
    unsigned long current_output_tiles = (unsigned long)(P+1) * 4 * 4 * num_filters;

    float *transformed_image = NULL;
    float *transformed_filter = NULL;
    float *gemm_output = NULL;

    //Below flags stores the status of buffer allocated through non-mempool allocation(i.e. malloc)
    bool image_flag = false;
    bool filter_flag = false;
    bool output_flag = false;

    int zenLibPoolEnable = zenEnvObj.zenLibMemPoolEnable;
    ZenLibMemoryPool *zenLibPoolBuffer;

    //ZenLibMemPool Optimization reuse tmp buffers from the pool. By default
    //  its enabled, export ZENDNN_MEMPOOL_ENABLE=0 will disable memory
    //  pool optimization
    //  Cases where buffers in pool are not free or requested size is more
    //  than available buffer size in Pool, control will fall back to
    //  default way of allocation
    if (zenLibPoolEnable) {
        zenLibPoolBuffer = ZenLibMemoryPool::getZenLibMemPool(0);
        if (zenLibPoolBuffer) {

            int status = zenLibPoolBuffer->acquireZenLibPoolBuf(&transformed_image,
                         current_image_tiles * sizeof(float),
                         1);
            if (status) {
                transformed_image = (float *)malloc(current_image_tiles * sizeof(
                                                        float));
                image_flag = true;
            }
            status = zenLibPoolBuffer->acquireZenLibPoolBuf(&transformed_filter,
                     current_filter_tiles * sizeof(float),
                     1);
            if (status) {
                transformed_filter = (float *)malloc(current_filter_tiles * sizeof(
                        float));
                filter_flag = true;
            }
            status = zenLibPoolBuffer->acquireZenLibPoolBuf(&gemm_output,
                     current_output_tiles * sizeof(float),
                     1);
            if (status) {
                gemm_output = (float *)malloc(current_output_tiles * sizeof(
                                                  float));
                output_flag = true;
            }
            if (!transformed_image || !transformed_filter || !gemm_output) {
                zendnnError(ZENDNN_ALGOLOG,
                            "winograd_2x2_3x3 Memory Error while allocating transformed_image or transformed_filter or gemm_output");

                if (transformed_image) {
                    free(transformed_image);
                }
                if (transformed_filter) {
                    free(transformed_filter);
                }
                if (gemm_output) {
                    free(gemm_output);
                }
                assert(0);
            }
        }
        else {
            zenLibPoolEnable = false;
        }
    }
    if (!zenLibPoolEnable) {

        static float *tmp_image = (float *)malloc(MAX_IMAGE_TILES * sizeof(
                                      float));
        static float *tmp_filter = (float *)malloc(MAX_FILTER_TILES * sizeof(
                                       float));
        static float *tmp_gemm_output = (float *)malloc(MAX_OUTPUT_TILES * sizeof(
                                            float));

        // check if current layer requires more memory
        // if it does, realloc
        if (current_image_tiles > MAX_IMAGE_TILES) {
            MAX_IMAGE_TILES = current_image_tiles;
            tmp_image = (float *)realloc(tmp_image,
                                         MAX_IMAGE_TILES * sizeof(float));
        }
        if (current_filter_tiles > MAX_FILTER_TILES) {
            MAX_FILTER_TILES = current_filter_tiles;
            tmp_filter = (float *)realloc(tmp_filter,
                                          MAX_FILTER_TILES * sizeof(float));
        }
        if (current_output_tiles > MAX_OUTPUT_TILES) {
            MAX_OUTPUT_TILES = current_output_tiles;
            tmp_gemm_output = (float *)realloc(tmp_gemm_output,
                                               MAX_OUTPUT_TILES * sizeof(float));
        }

        if (!tmp_image || !tmp_filter || !tmp_gemm_output) {
            zendnnError(ZENDNN_ALGOLOG,
                        "winograd_2x2_3x3 Memory Error while allocating transformed_image or transformed_filter or gemm_output");

            if (tmp_image) {
                free(tmp_image);
            }
            if (tmp_filter) {
                free(tmp_filter);
            }
            if (tmp_gemm_output) {
                free(tmp_gemm_output);
            }
            assert(0);
        }
        else {
            transformed_image = tmp_image;
            transformed_filter = tmp_filter;
            gemm_output = tmp_gemm_output;
        }
    }


    int d1,d2,d3,d4;
    auto start = std::chrono::high_resolution_clock::now();
    filter_transform_2x2_3x3(zenEnvObj, filter, num_channels, num_filters,
                             transformed_filter);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>
                       (duration);
    d1 = duration_ms.count();

    start = std::chrono::high_resolution_clock::now();
    input_transform_2x2_3x3(zenEnvObj, in_layer, num_images, height, width,
                            num_channels,
                            pad_t, pad_l, pad_b, pad_r,
                            transformed_image, P, out_height, out_width);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    d2 = duration_ms.count();

    start = std::chrono::high_resolution_clock::now();
    batched_gemm_2x2_3x3(zenEnvObj, transformed_image, P, num_channels, num_images,
                         transformed_filter, num_filters, gemm_output);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    d3 = duration_ms.count();

    start = std::chrono::high_resolution_clock::now();

    out_transform_2x2_3x3(zenEnvObj, gemm_output, P, num_filters,
                          out_layer, num_images, out_height, out_width, sum_fused);

    post_conv_transform(num_images, out_height, out_width, num_filters,
                        out_layer,
                        bias, relu, scale);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    d4 = duration_ms.count();

    int total = d1+d2+d3+d4;

    zendnnInfo(ZENDNN_ALGOLOG, "winograd_2x2_3x3, no_of_images=", num_images,
               " channels=", num_channels, " height=", height, " width=", width,
               " no_of_filter=", num_filters, " kernel_h=", kernel_h, " kernel_w=", kernel_w,
               " pad_t=", pad_t, " pad_b=", pad_b, " pad_l=", pad_l, " pad_r=", pad_r,
               " Time=", total, "ms",
               " Filter transform time =", 100.0f * d1/total, "%",
               " Input transform time =", 100.0f*d2/total,"%",
               " Gemm time =", 100.0f*d3/total, "%",
               " Output transform time =", 100.0f*d4/total,"%");


    //If ZenMemPool Optimization is enabled(default), update the state of
    //  Memory pool based on input_array address
    if (zenLibPoolEnable) {

        if (image_flag) {
            free(transformed_image);
        }
        else {
            zenLibPoolBuffer->zenLibMemPoolFree((float *)transformed_image);
        }

        if (filter_flag) {
            free(transformed_filter);
        }
        else {
            zenLibPoolBuffer->zenLibMemPoolFree((float *)transformed_filter);
        }

        if (output_flag) {
            free(gemm_output);
        }
        else {
            zenLibPoolBuffer->zenLibMemPoolFree((float *)gemm_output);
        }
    }


}
