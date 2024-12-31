/*******************************************************************************
* Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <unordered_map>
#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <time.h>
#include "common/zendnn_private.hpp"
#include "zendnn_logging.hpp"
#include "common/utils.hpp"
#include "common/zendnn_private.hpp"
#include "common/type_helpers.hpp"

#ifdef ZENDNN_ENABLE_LPGEMM_CONV
    #include "cpu/x64/zendnn_lpgemm_utils.hpp"
#endif

using namespace zendnn;

extern int graph_exe_count;

//Skip Iterations for auto tuner
//Can be set by environment variable ZENDNN_LPGEMM_SKIP_ITER
#define LPGEMM_SKIP_ITER_V1 4
#define LPGEMM_SKIP_ITER_V2 4
#define LPGEMM_SKIP_ITER_V3 10

//Evaluate iterations for auto tuner
//Can be set by environment variable ZENDNN_LPGEMM_EVALUATE_ITER
#define LPGEMM_EVALUATE_ITER_V1 6
#define LPGEMM_EVALUATE_ITER_V2 6
#define LPGEMM_EVALUATE_ITER_V3 10

//structure to make key
struct Key_lpgemm {
    int no_of_images;
    const int8_t *filter;
    int channels;
    int height;
    int width;
    int no_of_filter;
    int kernel_h;
    int kernel_w;
    int pad_t;
    int pad_l;
    int pad_b;
    int pad_r;
    int stride_h;
    int stride_w;
    int thread_count;

    bool operator==(const Key_lpgemm &other) const {
        return (thread_count == other.thread_count
                && no_of_images == other.no_of_images
                && filter == other.filter
                && channels == other.channels
                && height == other.height
                && width == other.width
                && no_of_filter == other.no_of_filter
                && kernel_h == other.kernel_h
                && kernel_w == other.kernel_w
                && pad_t == other.pad_t
                && pad_l == other.pad_l
                && pad_b == other.pad_b
                && pad_r == other.pad_r
                && stride_h == other.stride_h
                && stride_w == other.stride_w
               );
    }
};

namespace std {

template <>
struct hash<Key_lpgemm> {
    std::size_t operator()(const Key_lpgemm &k) const {
        std::size_t seed = 0;
        seed = zendnn::impl::hash_combine(seed, (k.no_of_images));
        seed = zendnn::impl::hash_combine(seed, (k.filter));
        seed = zendnn::impl::hash_combine(seed, (k.channels));
        seed = zendnn::impl::hash_combine(seed, (k.height));
        seed = zendnn::impl::hash_combine(seed, (k.width));
        seed = zendnn::impl::hash_combine(seed, (k.no_of_filter));
        seed = zendnn::impl::hash_combine(seed, (k.kernel_h));
        seed = zendnn::impl::hash_combine(seed, (k.kernel_w));
        seed = zendnn::impl::hash_combine(seed, (k.pad_t));
        seed = zendnn::impl::hash_combine(seed, (k.pad_l));
        seed = zendnn::impl::hash_combine(seed, (k.pad_b));
        seed = zendnn::impl::hash_combine(seed, (k.pad_r));
        seed = zendnn::impl::hash_combine(seed, (k.stride_h));
        seed = zendnn::impl::hash_combine(seed, (k.stride_w));
        seed = zendnn::impl::hash_combine(seed, (k.thread_count));
        return seed;
    }
};
}

enum convolution_os8 {
    convolution_gemm_u8s8s32os8 = 1,
    convolution_gemm_u8s8s16os8 = 2,
    convolution_ref_direct_os8 = 3,
    num_of_lpgemm_os8
};

enum convolution_os32 {
    convolution_gemm_u8s8s32os32 = 1,
    convolution_ref_direct_os32 = 2,
    num_of_lpgemm_os32
};

//Simplified Map having Key as struct and value as Algo.
std::unordered_map<Key_lpgemm, unsigned int>
conv_kernel_map;

//Map value is tuple of (iteration count, execution time of algo, Algo Path)
//Used in auto_compute_conv_v1 and auto_compute_conv_v3
std::unordered_map<Key_lpgemm,std::tuple<unsigned int, float, unsigned int>>
        conv_kernel_map1_helper;

//Map value is tuple of (vector<pair>, execution time of algo, Algo Path)
//Each element of vector represents pair of
//iteration count and average time for each algo(iteration count, average time)
std::unordered_map<Key_lpgemm,std::tuple<std::vector<std::pair<unsigned int,float>>, float, unsigned int>>
        conv_kernel_map2_helper;

/*Verion 1

  Works with framework (graph_exe_count)
  key = Key_lpgemm
  value = < count, time, algo >

  count : Total number of times the unique layer ran.
  time : Best time for selected algorithm used in MAP Creation Phase
  algo : The Algo to use.
*/

//Runs when ZENDNN_LPGEMM_AUTO_VERSION=1
//Makes use of graph_exe_count that is incremented by framework.
//Evaluates each algo during warmup evaluation phase.
//During evaluation, the best timed algo is selected.

int auto_compute_conv_v1(
    Key_lpgemm key_obj,
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
) {
    unsigned int selected_algo;
    unsigned int num_of_lpgemm_algo = supportedPath == 0 ? static_cast<int>
                                      (num_of_lpgemm_os8)-1 : static_cast<int>(num_of_lpgemm_os32)-1 ;
    float cur_algo_time; //current algorithm's execution time
    struct timeval start_n, end_n;

    //Number of iterations to run without creating map for each unique layer.
    int skip_iteration = zendnn::zendnn_getenv_int("ZENDNN_LPGEMM_SKIP_ITER",
                         LPGEMM_SKIP_ITER_V1);

    //Number of iterations to run for creating map for each layer.
    int evaluate_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_LPGEMM_EVALUATE_ITER",
                                  LPGEMM_EVALUATE_ITER_V1);

    //finds object in map
    auto found_obj = conv_kernel_map1_helper.find(key_obj);

    //If iteration count is less than Skip iteration then run default algo
    if (found_obj == conv_kernel_map1_helper.end() ||
            graph_exe_count < skip_iteration) {
        //Set algo initially
        selected_algo = supportedPath == 0 ? static_cast<int>
                        (convolution_os8::convolution_gemm_u8s8s32os8) : static_cast<int>
                        (convolution_os32::convolution_gemm_u8s8s32os32);

        //If Key not found in map then time the algo and add new element to map
        if (found_obj == conv_kernel_map1_helper.end()) {

            //Time start
#ifdef _WIN32
            auto start_n = std::chrono::high_resolution_clock::now();
#else
            gettimeofday(&start_n, 0);
#endif

            zendnnConvolutionLPGEMM(supportedPath, selected_algo,in_layer,
                                    no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                    pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                    out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                    output_scales,
                                    zero_point_dst, scale_count);
            //Time end
#ifdef _WIN32
            auto end_n = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> difference = end_n - start_n;
            cur_algo_time = difference.count();
#else
            gettimeofday(&end_n, 0);
            cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f +
                            (end_n.tv_usec - start_n.tv_usec)/ 1000.0f; //time in milliseconds
#endif

            //Create new entry
            conv_kernel_map[key_obj] = supportedPath == 0 ? static_cast<int>
                                       (convolution_os8::convolution_gemm_u8s8s32os8) : static_cast<int>
                                       (convolution_os32::convolution_gemm_u8s8s32os32);
            conv_kernel_map1_helper[key_obj] = {0, cur_algo_time, supportedPath == 0 ? static_cast<int>(convolution_os8::convolution_gemm_u8s8s32os8) : static_cast<int>(convolution_os32::convolution_gemm_u8s8s32os32)}; // {eval_count, time, algo}
        }
        else {
            zendnnConvolutionLPGEMM(supportedPath, selected_algo,in_layer,
                                    no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                    pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                    out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                    output_scales,
                                    zero_point_dst, scale_count);
        }
    }

    //Read Value from map.
    //Runs after skip iterations and evaluation iterations are done.
    else if (graph_exe_count >= evaluate_iteration + skip_iteration) {
        //Get best algo for given layer from MAP
        selected_algo = conv_kernel_map[key_obj];

        zendnnConvolutionLPGEMM(supportedPath, selected_algo,in_layer,
                                no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                output_scales,
                                zero_point_dst, scale_count);
    }
    //Runs for evaluate iterations
    //Updates the map values accordingly
    else {
        //Get the number of iteration already ran and select Algo to run for current iteration
        //get<0>(found_obj->second) = count of iteration
        selected_algo = (std::get<0>(found_obj->second)%num_of_lpgemm_algo)+1;
        std::get<0>(found_obj->second) += 1;

        //timer start
#ifdef _WIN32
        auto start_n = std::chrono::high_resolution_clock::now();
#else
        gettimeofday(&start_n, 0);
#endif

        zendnnConvolutionLPGEMM(supportedPath, selected_algo,in_layer,
                                no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                output_scales,
                                zero_point_dst, scale_count);

        //timer end
#ifdef _WIN32
        auto end_n = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> difference = end_n - start_n;
        cur_algo_time = difference.count();
#else
        gettimeofday(&end_n, 0);
        cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f +
                        (end_n.tv_usec - start_n.tv_usec)/ 1000.0f; //time in milliseconds
#endif

        //If current run gives better timing then update
        if (cur_algo_time < std::get<1>(found_obj->second)) {
            std::get<1>(found_obj->second) = cur_algo_time; //Minimum time for chosen algo
            std::get<2>(found_obj->second) = selected_algo;
            conv_kernel_map[key_obj] = selected_algo;
        }
    }
    //return zenEnvObj.zenGEMMalgo;
    return selected_algo;
}


/*Verion 2

  Works with framework(graph_exe_count)
  key = Key_lpgemm
  value = < vector<iteration count,time>, time, algo >

  vector<count, time> : Iteration count of each algo and their average time.
  time : Best time for selected algorithm used in MAP Creation Phase.
  algo : The Algo to use.
*/

//Runs when ZENDNN_LPGEMM_AUTO_VERSION=2
//Makes use of graph_exe_count that is incremented by framework.
//For each graph_exe_count one of the algorithm runs
//Evaluates each algo during evaluation phase based on average time.
int auto_compute_conv_v2(
    Key_lpgemm key_obj,
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
) {
    unsigned int selected_algo;
    unsigned int num_of_lpgemm_algo = supportedPath == 0 ? static_cast<int>
                                      (num_of_lpgemm_os8)-1 :static_cast<int>(num_of_lpgemm_os32)-1 ;
    float cur_algo_time; //current algorithm's execution time
    struct timeval start_n, end_n;

    //Number of iterations to run without creating map for each unique layer.
    int skip_iteration = zendnn::zendnn_getenv_int("ZENDNN_LPGEMM_SKIP_ITER",
                         LPGEMM_SKIP_ITER_V2);

    //Number of iterations to run for creating map for each layer.
    int evaluate_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_LPGEMM_EVALUATE_ITER",
                                  LPGEMM_EVALUATE_ITER_V2);

    //finds object in map
    auto found_obj = conv_kernel_map2_helper.find(key_obj);

    //If iteration count is less than Skip iteration then run default algo
    if (found_obj == conv_kernel_map2_helper.end() ||
            graph_exe_count < skip_iteration) {
        //Set algo initially for skip runs
        selected_algo = supportedPath == 0 ? static_cast<int>
                        (convolution_os8::convolution_gemm_u8s8s32os8) :static_cast<int>
                        (convolution_os32::convolution_gemm_u8s8s32os32);

        //If Key not found in map then time the algo and add new element to map
        if (found_obj == conv_kernel_map2_helper.end()) {
            //Time start
#ifdef _WIN32
            auto start_n = std::chrono::high_resolution_clock::now();
#else
            gettimeofday(&start_n, 0);
#endif

            zendnnConvolutionLPGEMM(supportedPath, selected_algo,in_layer,
                                    no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                    pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                    out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                    output_scales,
                                    zero_point_dst, scale_count);

            //Time end
#ifdef _WIN32
            auto end_n = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> difference = end_n - start_n;
            cur_algo_time = difference.count();
#else
            gettimeofday(&end_n, 0);
            cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f +
                            (end_n.tv_usec - start_n.tv_usec) / 1000.0f; //time in milliseconds
#endif

            //Create new entry
            //initial vector for average time and iteration count for each algorithms.
            std::vector<std::pair<unsigned int,float>> initial_vec(num_of_lpgemm_algo, {0,0.0});
            conv_kernel_map2_helper[key_obj] = {initial_vec, cur_algo_time, zenMatMulAlgoType::MATMUL_BLOCKED_JIT_FP32};
            conv_kernel_map[key_obj] = supportedPath == 0 ? static_cast<int>
                                       (convolution_os8::convolution_gemm_u8s8s32os8) :static_cast<int>
                                       (convolution_os32::convolution_gemm_u8s8s32os32);
            //value of map {vector<{iteration,avg time}>, time, algo}
        }
        else {
            zendnnConvolutionLPGEMM(supportedPath, selected_algo,in_layer,
                                    no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                    pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                    out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                    output_scales,
                                    zero_point_dst, scale_count);
        }
    }
    //Read Value from map.
    //Runs after skip iterations and evaluation iterations are done.
    else if (graph_exe_count >= evaluate_iteration + skip_iteration) {
        //Get best algo for given layer from MAP (tuple's 2nd index has algo)
        selected_algo = conv_kernel_map[key_obj];
        zendnnConvolutionLPGEMM(supportedPath, selected_algo,in_layer,
                                no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                output_scales,
                                zero_point_dst, scale_count);
    }
    //Runs for evaluate iterations
    //Updates the map values accordingly
    else {
        //Run single algorithm for each graph_exe_count value.
        selected_algo = ((graph_exe_count-skip_iteration)%num_of_lpgemm_algo) +1;

        //timer start
#ifdef _WIN32
        auto start_n = std::chrono::high_resolution_clock::now();
#else
        gettimeofday(&start_n, 0);
#endif

        zendnnConvolutionLPGEMM(supportedPath, selected_algo,in_layer,
                                no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                output_scales,
                                zero_point_dst, scale_count);

        //timer end
#ifdef _WIN32
        auto end_n = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> difference = end_n - start_n;
        cur_algo_time = difference.count();
#else
        gettimeofday(&end_n, 0);
        cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f +
                        (end_n.tv_usec - start_n.tv_usec) / 1000.0f; //time in milliseconds
#endif

        //Finding the current algorithm's average time and iteration stored in Map
        float t_algo =
            std::get<0>(found_obj->second)[selected_algo].second;
        int i_algo =
            std::get<0>(found_obj->second)[selected_algo].first;

        //updating the average time and iteration for the current algorithm run.
        cur_algo_time = ((t_algo*i_algo) + cur_algo_time)/(i_algo+1);
        std::get<0>(found_obj->second)[selected_algo].second =
            cur_algo_time;
        std::get<0>(found_obj->second)[selected_algo].first +=1;

        //If current run gives better timing then update
        if (cur_algo_time < std::get<1>(found_obj->second)) {
            std::get<1>(found_obj->second) = cur_algo_time; //Minimum time for chosen algo
            std::get<2>(found_obj->second) =
                selected_algo; //Algo with minimum time (1-NUM_OF_LPGEMM_ALGO)
            conv_kernel_map[key_obj] = selected_algo;//zenEnvObj.zenGEMMalgo;
        }
    }

    return selected_algo;//zenEnvObj.zenGEMMalgo;
}



/*verion 3
  Makes use of eval_count to decide when to fetch value from map.
  Uses iteration count of each unique layer.
  Doesn't need framework to increment the graph_exe_count.

  Map value tuple
  <
    iteration_count,
    time,
    algo
  >
*/
//Runs when ZENDNN_LPGEMM_AUTO_VERSION=3
int auto_compute_conv_v3(
    Key_lpgemm key_obj,
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
) {
    unsigned int selected_algo;
    unsigned int num_of_lpgemm_algo = supportedPath == 0 ? static_cast<int>
                                      (num_of_lpgemm_os8)-1 :static_cast<int>(num_of_lpgemm_os32)-1 ;
    float cur_algo_time; //current algorithm's execution time
    struct timeval start_n, end_n;

    //Number of iterations to run without creating map for each unique layer.
    int skip_iteration = zendnn::zendnn_getenv_int("ZENDNN_LPGEMM_SKIP_ITER",
                         LPGEMM_SKIP_ITER_V3);

    //Number of iterations to run for creating map for each layer.
    int evaluate_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_LPGEMM_EVALUATE_ITER",
                                  LPGEMM_EVALUATE_ITER_V3);

    //finds object in map
    auto found_obj = conv_kernel_map1_helper.find(key_obj);

    //If current iterations less than Skip iteration then run default algo.
    //Checks using the (0) element of map value that denotes count of iterations in map
    if (found_obj == conv_kernel_map1_helper.end() ||
            std::get<0>(found_obj->second) < skip_iteration) {

        //Set algo 3 initially
        selected_algo = supportedPath == 0 ? static_cast<int>
                        (convolution_os8::convolution_gemm_u8s8s32os8) :static_cast<int>
                        (convolution_os32::convolution_gemm_u8s8s32os32);

        //If Key not found in map then time the algo and add new element to map
        if (found_obj == conv_kernel_map1_helper.end()) {

            //Time start
#ifdef _WIN32
            auto start_n = std::chrono::high_resolution_clock::now();
#else
            gettimeofday(&start_n, 0);
#endif
            zendnnConvolutionLPGEMM(supportedPath, selected_algo,in_layer,
                                    no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                    pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                    out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                    output_scales,
                                    zero_point_dst, scale_count);

            //Time end
#ifdef _WIN32
            auto end_n = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> difference = end_n - start_n;
            cur_algo_time = difference.count();
#else
            gettimeofday(&end_n, 0);
            cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f
                            + (end_n.tv_usec - start_n.tv_usec)/1000.0f; //time in milliseconds
#endif

            //Create new entry
            conv_kernel_map1_helper[key_obj] = {1, cur_algo_time, supportedPath == 0 ? static_cast<int>(convolution_os8::convolution_gemm_u8s8s32os8) :static_cast<int>(convolution_os32::convolution_gemm_u8s8s32os32)}; // {iter_count, time, algo}
            conv_kernel_map[key_obj] = supportedPath == 0 ? static_cast<int>
                                       (convolution_os8::convolution_gemm_u8s8s32os8) : static_cast<int>
                                       (convolution_os32::convolution_gemm_u8s8s32os32);
        }
        //If key found then increment the iter_count and run algo 3.
        else {
            std::get<0>(found_obj->second) += 1;
            zendnnConvolutionLPGEMM(supportedPath, selected_algo,in_layer,
                                    no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                    pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                    out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                    output_scales,
                                    zero_point_dst, scale_count);
        }
    }
    //Read Value from map.
    //Runs after skip iterations and evaluation iterations are done.
    else if (std::get<0>(found_obj->second) > evaluate_iteration + skip_iteration) {
        //Get best algo for given layer from MAP
        selected_algo = conv_kernel_map[key_obj];
        zendnnConvolutionLPGEMM(supportedPath, selected_algo,in_layer,
                                no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                output_scales,
                                zero_point_dst, scale_count);
    }
    //Updates the map values by running different algorithms
    else {
        //Get the number of iteration already ran and select Algo to run for current iteration
        //get<0>(found_obj->second) = count of iteration
        selected_algo = (std::get<0>(found_obj->second)%num_of_lpgemm_algo) +1;
        std::get<0>(found_obj->second) += 1;
        //timer start
#ifdef _WIN32
        auto start_n = std::chrono::high_resolution_clock::now();
#else
        gettimeofday(&start_n, 0);
#endif

        zendnnConvolutionLPGEMM(supportedPath, selected_algo,in_layer,
                                no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                output_scales,
                                zero_point_dst, scale_count);

        //timer end
#ifdef _WIN32
        auto end_n = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> difference = end_n - start_n;
        cur_algo_time = difference.count();
#else
        gettimeofday(&end_n, 0);
        cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f +
                        (end_n.tv_usec - start_n.tv_usec)/ 1000.0f; //time in milliseconds
#endif

        //If current run gives better timing then update
        if (cur_algo_time < std::get<1>(found_obj->second)) {
            std::get<1>(found_obj->second) = cur_algo_time; //Minimum time for chosen algo
            std::get<2>(found_obj->second)
                =selected_algo; //Algo with minimum time (1-NUM_OF_LPGEMM_ALGO)
            conv_kernel_map[key_obj] = selected_algo;
        }
    }

    return selected_algo;
}


//This is the wrapper function
//It calls appropriate version of auto_tuner.
//Currently autotuner supports os8 and os32 with input uint8
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
) {
    unsigned int algo_type;

    Key_lpgemm key_obj;

    //Select auto_tuner version
    //Auto_type 1 and 2 works only with framework.
    unsigned int auto_type = zendnn::zendnn_getenv_int("ZENDNN_LPGEMM_AUTO_VERSION",
                             1);

    //It is used to know if weights address should be enabled or not for map.
    //0: disable, 1: enable.
    int map_type = zendnn::zendnn_getenv_int("ZENDNN_LPGEMM_MAP_TYPE",1);

    key_obj.no_of_images = no_of_images;
    key_obj.channels = channels;
    key_obj.height = height;
    key_obj.width = width;
    key_obj.no_of_filter = no_of_filter;
    key_obj.kernel_h = kernel_h;
    key_obj.kernel_w = kernel_w;
    key_obj.pad_t = pad_t;
    key_obj.pad_l = pad_l;
    key_obj.pad_b = pad_b;
    key_obj.pad_r = pad_r;
    key_obj.stride_h = stride_h;
    key_obj.stride_w = stride_w;

    //This condition makes sure that address
    //doesn't gets saved if map_type is NOT 1.
    key_obj.filter = map_type == 1 ? filter : NULL;
    key_obj.thread_count = zendnn::zendnn_getenv_int("OMP_NUM_THREADS",1);

    //If graph_exe_count is incremented by framework
    if (graph_exe_count != -1) {
        //uses framework.
        if (auto_type == 1) {
            algo_type = auto_compute_conv_v1(key_obj, supportedPath, in_layer,
                                             no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                             pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                             out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                             output_scales,
                                             zero_point_dst, scale_count);
        }
        //uses framework(graph_exe_count) and average time
        else if (auto_type == 2) {
            algo_type = auto_compute_conv_v2(key_obj, supportedPath, in_layer,
                                             no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                             pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                             out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                             output_scales,
                                             zero_point_dst, scale_count);
        }
        //Without framework(graph_exe_count)
        else {
            algo_type = auto_compute_conv_v3(key_obj, supportedPath, in_layer,
                                             no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                             pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                             out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                             output_scales,
                                             zero_point_dst, scale_count);
        }
    }

    //When framework doesn't increment graph_exe_count
    else {
        algo_type = auto_compute_conv_v3(key_obj, supportedPath, in_layer,
                                         no_of_images, channels, height, width, filter, no_of_filter, kernel_h, kernel_w,
                                         pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, bias, out_layer, out_height,
                                         out_width, concat, filter_offset, total_filters, reluFused, elementwiseType,
                                         output_scales,
                                         zero_point_dst, scale_count);
    }

    return algo_type;
}
