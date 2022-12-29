/*******************************************************************************
* Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <zendnn_helper.hpp>
#include <time.h>

#include "utils.hpp"

using namespace zendnn;

//Total num of algo available
#define NUM_OF_ALGO 4

//Skip Iterations for auto tuner
//Can be set by environment variable ZENDNN_MATMUL_SKIP_ITER
#define MATMUL_SKIP_ITER_V1 4
#define MATMUL_SKIP_ITER_V2 4
#define MATMUL_SKIP_ITER_V3 10

//Evaluate iterations for auto tuner
//Can be set by environment variable ZENDNN_MATMUL_EVALUATE_ITER
#define MATMUL_EVALUATE_ITER_V1 6
#define MATMUL_EVALUATE_ITER_V2 6
#define MATMUL_EVALUATE_ITER_V3 10

//This tracks the no. of times graph executed
// from the framework.
unsigned int graph_exe_count = -1;

//structure to make key
struct Key_matmul {
    bool transpose_input;
    bool transpose_weights;
    unsigned int m;
    unsigned int k;
    unsigned int n;
    unsigned int lda;
    unsigned int ldb;
    unsigned int ldc;
    unsigned int thread_count;
    const void *weights;

    bool operator==(const Key_matmul &other) const {
        return (thread_count == other.thread_count
                && m == other.m
                && k == other.k
                && n == other.n
                && lda == other.lda
                && ldb == other.ldb
                && ldc == other.ldc
                && weights == other.weights
                && transpose_input == other.transpose_input
                && transpose_weights == other.transpose_weights
               );
    }
};

namespace std {

template <>
struct hash<Key_matmul> {
    std::size_t operator()(const Key_matmul &k) const {
        std::size_t seed = 0;
        seed = zendnn::impl::hash_combine(seed, (k.transpose_input));
        seed = zendnn::impl::hash_combine(seed, (k.transpose_weights));
        seed = zendnn::impl::hash_combine(seed, (k.m));
        seed = zendnn::impl::hash_combine(seed, (k.k));
        seed = zendnn::impl::hash_combine(seed, (k.n));
        seed = zendnn::impl::hash_combine(seed, (k.lda));
        seed = zendnn::impl::hash_combine(seed, (k.ldb));
        seed = zendnn::impl::hash_combine(seed, (k.ldc));
        seed = zendnn::impl::hash_combine(seed, (k.thread_count));
        seed = zendnn::impl::hash_combine(seed, (k.weights));
        return seed;
    }
};
}


//Simplified Map having Key as struct and value as Algo.
std::unordered_map<Key_matmul, unsigned int>
matmul_kernel_map;

//Map value is tuple of (iteration count, execution time of algo, Algo Path)
//Used in auto_compute_matmul_v1 and auto_compute_matmul_v3
std::unordered_map<Key_matmul,std::tuple<unsigned int, float, unsigned int>>
        matmul_kernel_map1_helper;

//Map value is tuple of (vector<pair>, execution time of algo, Algo Path)
//Each element of vector represents pair of
//iteration count and average time for each algo(iteration count, average time)
std::unordered_map<Key_matmul,std::tuple<std::vector<std::pair<unsigned int,float>>, float, unsigned int>>
        matmul_kernel_map2_helper;



/*Verion 1

  Works with framework (graph_exe_count)
  key = Key_matmul
  value = < count, time, algo >

  count : Total number of times the unique layer ran.
  time : Best time for selected algorithm used in MAP Creation Phase
  algo : The Algo to use.
*/

//Runs when ZENDNN_GEMM_AUTO_TYPE=1
//Makes use of graph_exe_count that is incremented by framework.
//Evaluates each algo during warmup(skip + evaluation) phase.
//During evaluation, the best timed algo is selected.
int auto_compute_matmul_v1(
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
    const float *weights,
    const int ldb,
    const float *bias,
    const bool relu,
    const int gelu,
    const float beta,
    float *output,
    const int ldc
) {

    float cur_algo_time; //current algorithm's execution time
    struct timeval start_n, end_n;

    Key_matmul key_obj;

    //It is used here to know if weights should be enabled or not for map.
    int map_type = zendnn::zendnn_getenv_int("ZENDNN_GEMM_MAP_TYPE",1);

    //Number of iterations to run without creating map for each unique layer.
    int skip_iteration = zendnn::zendnn_getenv_int("ZENDNN_MATMUL_SKIP_ITER",
                         MATMUL_SKIP_ITER_V1);

    //Number of iterations to run for creating map for each layer.
    int evaluate_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_MATMUL_EVALUATE_ITER",
                                  MATMUL_EVALUATE_ITER_V1);

    key_obj.transpose_input = transpose_input;
    key_obj.transpose_weights = transpose_weights;
    key_obj.m = m;
    key_obj.k = k;
    key_obj.n = n;
    key_obj.lda = lda;
    key_obj.ldb = ldb;
    key_obj.ldc = ldc;
    key_obj.weights = map_type==2 ? weights : NULL;
    key_obj.thread_count = zenEnvObj.omp_num_threads;

    //finds object in map
    auto found_obj = matmul_kernel_map1_helper.find(key_obj);

    //If iteration count is less than Skip iteration then run default algo
    if (found_obj == matmul_kernel_map1_helper.end() ||
            graph_exe_count < skip_iteration) {

        //Set algo 3 initially
        zenEnvObj.zenGEMMalgo = zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1;

        //If Key not found in map then time the algo and add new element to map
        if (found_obj == matmul_kernel_map1_helper.end()) {

            //Time start
#ifdef _WIN32
            auto start_n = std::chrono::high_resolution_clock::now();
#else
            gettimeofday(&start_n, 0);
#endif

            zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_weights, m,
                           k,
                           n,
                           alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);

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
            matmul_kernel_map[key_obj] = zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1;
            matmul_kernel_map1_helper[key_obj] = {0, cur_algo_time, 3}; // {eval_count, time, algo}
        }
        else {
            zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_weights, m,
                           k,
                           n,
                           alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);
        }
    }

    //Read Value from map.
    //Runs after skip iterations and evaluation iterations are done.
    else if (graph_exe_count >= evaluate_iteration + skip_iteration) {

        //Get best algo for given layer from MAP
        zenEnvObj.zenGEMMalgo = matmul_kernel_map[key_obj];
        zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_weights, m,
                       k,
                       n,
                       alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);
    }
    //Runs for evaluate iterations
    //Updates the map values accordingly
    else {

        //Get the number of iteration already ran and select Algo to run for current iteration
        //get<0>(found_obj->second) = count of iteration
        zenEnvObj.zenGEMMalgo = (std::get<0>(found_obj->second)%NUM_OF_ALGO) +1;
        std::get<0>(found_obj->second) += 1;

        //timer start
#ifdef _WIN32
        auto start_n = std::chrono::high_resolution_clock::now();
#else
        gettimeofday(&start_n, 0);
#endif

        zenMatMul_gemm(zenEnvObj, true, Layout,transpose_input, transpose_weights, m, k,
                       n,
                       alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);
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
            std::get<2>(found_obj->second) =
                zenEnvObj.zenGEMMalgo; //Algo with minimum time (1-4)

            matmul_kernel_map[key_obj] = zenEnvObj.zenGEMMalgo;
        }

    }

    return zenEnvObj.zenGEMMalgo;
}


/*Verion 2

  Works with framework(graph_exe_count)
  key = Key_matmul
  value = < vector<iteration count,time>, time, algo >

  vector<count, time> : Iteration count of each algo and their average time.
  time : Best time for selected algorithm used in MAP Creation Phase.
  algo : The Algo to use.
*/

//Runs when ZENDNN_GEMM_AUTO_TYPE=2
//Makes use of graph_exe_count that is incremented by framework.
//For each graph_exe_count one of the algorithm runs
//Evaluates each algo during warmup(skip + evaluation) phase based on average time.
int auto_compute_matmul_v2(
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
    const float *weights,
    const int ldb,
    const float *bias,
    const bool relu,
    const int gelu,
    const float beta,
    float *output,
    const int ldc
) {

    float cur_algo_time; //current algorithm's execution time
    struct timeval start_n, end_n;

    Key_matmul key_obj;

    //It is used here to know if weights should be enabled or not for map.
    int map_type = zendnn::zendnn_getenv_int("ZENDNN_GEMM_MAP_TYPE",1);

    //Number of iterations to run without creating map for each unique layer.
    int skip_iteration = zendnn::zendnn_getenv_int("ZENDNN_MATMUL_SKIP_ITER",
                         MATMUL_SKIP_ITER_V2);

    //Number of iterations to run for creating map for each layer.
    int evaluate_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_MATMUL_EVALUATE_ITER",
                                  MATMUL_EVALUATE_ITER_V2);

    key_obj.transpose_input = transpose_input;
    key_obj.transpose_weights = transpose_weights;
    key_obj.m = m;
    key_obj.k = k;
    key_obj.n = n;
    key_obj.lda = lda;
    key_obj.ldb = ldb;
    key_obj.ldc = ldc;
    key_obj.weights = map_type==2 ? weights : NULL;
    key_obj.thread_count = zenEnvObj.omp_num_threads;

    //finds object in map
    auto found_obj = matmul_kernel_map2_helper.find(key_obj);

    //If iteration count is less than Skip iteration then run default algo
    if (found_obj == matmul_kernel_map2_helper.end() ||
            graph_exe_count < skip_iteration) {

        //Set algo 3 initially
        zenEnvObj.zenGEMMalgo = zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1;

        //If Key not found in map then time the algo and add new element to map
        if (found_obj == matmul_kernel_map2_helper.end()) {

            //Time start
#ifdef _WIN32
            auto start_n = std::chrono::high_resolution_clock::now();
#else
            gettimeofday(&start_n, 0);
#endif

            zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_weights, m,
                           k,
                           n,
                           alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);

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
            std::vector<std::pair<unsigned int,float>> initial_vec(NUM_OF_ALGO, {0,0.0});
            matmul_kernel_map2_helper[key_obj] = {initial_vec, cur_algo_time, zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1};
            matmul_kernel_map[key_obj] = zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1;
            //value of map {vector<{iteration,avg time}>, time, algo}
        }
        else {
            zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_weights, m,
                           k,
                           n,
                           alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);
        }
    }
    //Read Value from map.
    //Runs after skip iterations and evaluation iterations are done.
    else if (graph_exe_count >= evaluate_iteration + skip_iteration) {

        //Get best algo for given layer from MAP (tuple's 2nd index has algo)
        zenEnvObj.zenGEMMalgo = matmul_kernel_map[key_obj];

        zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_weights, m,
                       k,
                       n,
                       alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);
    }
    //Runs for evaluate iterations
    //Updates the map values accordingly
    else {

        //Run single algorithm for each graph_exe_count value.
        zenEnvObj.zenGEMMalgo = ((graph_exe_count-skip_iteration)%NUM_OF_ALGO) +1;

        //timer start
#ifdef _WIN32
        auto start_n = std::chrono::high_resolution_clock::now();
#else
        gettimeofday(&start_n, 0);
#endif

        zenMatMul_gemm(zenEnvObj, true, Layout,transpose_input, transpose_weights, m, k,
                       n,
                       alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);
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
        float t_algo =  std::get<0>(found_obj->second)[zenEnvObj.zenGEMMalgo -
                                              1].second;
        int i_algo = std::get<0>(found_obj->second)[zenEnvObj.zenGEMMalgo - 1].first;

        //updating the average time and iteration for the current algorithm run.
        cur_algo_time = ((t_algo*i_algo) + cur_algo_time)/(i_algo+1);
        std::get<0>(found_obj->second)[zenEnvObj.zenGEMMalgo - 1].second =
            cur_algo_time;
        std::get<0>(found_obj->second)[zenEnvObj.zenGEMMalgo - 1].first +=1;

        //If current run gives better timing then update
        if (cur_algo_time < std::get<1>(found_obj->second)) {
            std::get<1>(found_obj->second) = cur_algo_time; //Minimum time for chosen algo
            std::get<2>(found_obj->second) =
                zenEnvObj.zenGEMMalgo; //Algo with minimum time (1-4)
            matmul_kernel_map[key_obj] = zenEnvObj.zenGEMMalgo;
        }
    }

    return zenEnvObj.zenGEMMalgo;
}



/*verion 3
  Makes use of eval_count to decide when to fetch value from map.
  Doesn't need framework to increment the graph_exe_count.

  Map value tuple
  <
    iteration_count,
    time,
    algo
  >
*/
//Runs when ZENDNN_GEMM_AUTO_TYPE=3
int auto_compute_matmul_v3(
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
    const float *weights,
    const int ldb,
    const float *bias,
    const bool relu,
    const int gelu,
    const float beta,
    float *output,
    const int ldc
) {

    float cur_algo_time; //current algorithm's execution time
    struct timeval start_n, end_n;

    Key_matmul key_obj;

    //It is used here to know if weights should be enabled or not for map.
    int map_type = zendnn::zendnn_getenv_int("ZENDNN_GEMM_MAP_TYPE",1);

    //Number of iterations to run without creating map for each unique layer.
    int skip_iteration = zendnn::zendnn_getenv_int("ZENDNN_MATMUL_SKIP_ITER",
                         MATMUL_SKIP_ITER_V3);

    //Number of iterations to run for creating map for each layer.
    int evaluate_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_MATMUL_EVALUATE_ITER",
                                  MATMUL_EVALUATE_ITER_V3);

    key_obj.transpose_input = transpose_input;
    key_obj.transpose_weights = transpose_weights;
    key_obj.m = m;
    key_obj.k = k;
    key_obj.n = n;
    key_obj.lda = lda;
    key_obj.ldb = ldb;
    key_obj.ldc = ldc;
    key_obj.weights = map_type==2 ? weights : NULL;
    key_obj.thread_count = zenEnvObj.omp_num_threads;

    //finds object in map
    auto found_obj = matmul_kernel_map1_helper.find(key_obj);

    //If current iterations less than Skip iteration then run default algo.
    //Checks using the (0) element of map value that denotes count of iterations in map
    if (found_obj == matmul_kernel_map1_helper.end() ||
            std::get<0>(found_obj->second) < skip_iteration) {

        //Set algo 3 initially
        zenEnvObj.zenGEMMalgo = zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1;

        //If Key not found in map then time the algo and add new element to map
        if (found_obj == matmul_kernel_map1_helper.end()) {

            //Time start
#ifdef _WIN32
            auto start_n = std::chrono::high_resolution_clock::now();
#else
            gettimeofday(&start_n, 0);
#endif

            zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_weights, m,
                           k,
                           n,
                           alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);

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
            matmul_kernel_map1_helper[key_obj] = {1, cur_algo_time, zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1}; // {iter_count, time, algo}
            matmul_kernel_map[key_obj] = zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1;
        }
        //If key found then increment the iter_count and run algo 3.
        else {
            std::get<0>(found_obj->second) += 1;
            zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_weights, m,
                           k,
                           n,
                           alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);
        }
    }
    //Read Value from map.
    //Runs after skip iterations and evaluation iterations are done.
    else if (std::get<0>(found_obj->second) > evaluate_iteration + skip_iteration) {

        //Get best algo for given layer from MAP
        zenEnvObj.zenGEMMalgo = matmul_kernel_map[key_obj];
        zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_weights, m,
                       k,
                       n,
                       alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);

    }
    //Updates the map values by running different algorithms
    else {

        //Get the number of iteration already ran and select Algo to run for current iteration
        //get<0>(found_obj->second) = count of iteration
        zenEnvObj.zenGEMMalgo = (std::get<0>(found_obj->second)%NUM_OF_ALGO) +1;
        std::get<0>(found_obj->second) += 1;
        //timer start
#ifdef _WIN32
        auto start_n = std::chrono::high_resolution_clock::now();
#else
        gettimeofday(&start_n, 0);
#endif

        zenMatMul_gemm(zenEnvObj, true, Layout,transpose_input, transpose_weights, m, k,
                       n,
                       alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);
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
            std::get<2>(found_obj->second) =
                zenEnvObj.zenGEMMalgo; //Algo with minimum time (1-4)
            matmul_kernel_map[key_obj] = zenEnvObj.zenGEMMalgo;
        }

    }

    return zenEnvObj.zenGEMMalgo;
}


//This is the wrapper function
//It calls appropriate version of auto_tuner.

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
    const float *weights,
    const int ldb,
    const float *bias,
    const bool relu,
    const int gelu,
    const float beta,
    float *output,
    const int ldc
) {

    unsigned int algo_type;

    //Select auto_tuner version
    //Auto_type 1 and 2 works only with framework.
    unsigned int auto_type = zendnn::zendnn_getenv_int("ZENDNN_GEMM_AUTO_TYPE",3);

    //If graph_exe_count is incremented by framework
    if (graph_exe_count != -1) {

        //uses framework.
        if (auto_type == 1) {
            algo_type = auto_compute_matmul_v1(zenEnvObj, Layout, transpose_input,
                                               transpose_weights,
                                               m, k, n, alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);
        }
        //uses framework(graph_exe_count) and average time
        else if (auto_type == 2) {
            algo_type = auto_compute_matmul_v2(zenEnvObj, Layout, transpose_input,
                                               transpose_weights,
                                               m, k, n, alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);
        }
        //Without framework(graph_exe_count)
        else {
            algo_type = auto_compute_matmul_v3(zenEnvObj, Layout, transpose_input,
                                               transpose_weights,
                                               m, k, n, alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);
        }
    }

    //When framework doesn't increment graph_exe_count
    else {
        algo_type = auto_compute_matmul_v3(zenEnvObj, Layout, transpose_input,
                                           transpose_weights,
                                           m, k, n, alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc);
    }

    return algo_type;
}
