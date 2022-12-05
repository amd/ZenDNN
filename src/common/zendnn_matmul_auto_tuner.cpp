/*******************************************************************************
* Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <unordered_map>
#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <zendnn_helper.hpp>
#include <time.h>
#include <sys/time.h>

using namespace zendnn;

//Total num of algo available
#define NUM_OF_ALGO 4

//Skip Iterations for auto tuner
//Can be set by environment variable ZENDNN_MATMUL_SKIP_ITER
#define MATMUL_SKIP_ITER_V2 2
#define MATMUL_SKIP_ITER_V3 4
#define MATMUL_SKIP_ITER_V4 10

//Evaluate iterations for auto tuner
//Can be set by environment variable ZENDNN_MATMUL_EVALUATE_ITER
#define MATMUL_EVALUATE_ITER_V1 2
#define MATMUL_EVALUATE_ITER_V3 6
#define MATMUL_EVALUATE_ITER_V4 10
//This tracks the no. of times graph executed
// from the framework.
unsigned int graph_exe_count = -1;

//structure to make key
struct Key_matmul {
    bool Layout;
    bool transpose_input;
    bool transpose_filter;
    unsigned int m;
    unsigned int k;
    unsigned int n;
    double alpha;
    unsigned int lda;
    unsigned int ldb;
    unsigned int ldc;
    unsigned int thread_count;

    bool operator==(const Key_matmul &other) const {
        return (thread_count == other.thread_count
                && m == other.m
                && k == other.k
                && n == other.n
                && alpha == other.alpha
                && lda == other.lda
                && ldb == other.ldb
                && ldc == other.ldc
                && Layout == other.Layout
                && transpose_input == other.transpose_input
                && transpose_filter == other.transpose_filter
               );
    }
};

//Finding hash for given seed.
static size_t hash_combine(size_t seed, const int &v) {
    return seed ^= std::hash<int> {}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std {

template <>
struct hash<Key_matmul> {
    std::size_t operator()(const Key_matmul &k) const {
        std::size_t seed = 0;
        seed = hash_combine(seed, (k.Layout));
        seed = hash_combine(seed, (k.transpose_input));
        seed = hash_combine(seed, (k.transpose_filter));
        seed = hash_combine(seed, (k.m));
        seed = hash_combine(seed, (k.k));
        seed = hash_combine(seed, (k.n));
        seed = hash_combine(seed, (k.alpha));
        seed = hash_combine(seed, (k.lda));
        seed = hash_combine(seed, (k.ldb));
        seed = hash_combine(seed, (k.ldc));
        seed = hash_combine(seed, (k.thread_count));

        return seed;
    }
};
}

//Map having Parameters as key and value as pair of (count,algo Path)
//Used in auto_compute_matmul_v1 and auto_compute_matmul_v2
std::unordered_map<Key_matmul,std::pair<unsigned int,int>>
        matmul_kernel_map;

//Map value is tuple of (iteration count, execution time of algo, Algo Path)
//Used in auto_compute_matmul_v3 and auto_compute_matmul_v4
std::unordered_map<Key_matmul,std::tuple<unsigned int, float, unsigned int>>
        matmul_kernel_map2;


//Auto tuner for matmul
//Runs only when ZENDNN_GEMM_ALGO=0
//For choosing best algo it runs all available algo
// using a tight FOR loop.
//Every EVALUATE_ITER it updates the best algo.
//Uses EVALUATE_ITER only.
int auto_compute_matmul_v1(
    zendnn::zendnnEnv zenEnvObj,
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
    const int ldc
) {

    int best_algo; //storing path for best kernel (1-4)
    float cur_min_time = -1; //storing minimum time of kernel
    float cur_algo_time = 0; //current algorithm's execution time
    struct timeval start_n, end_n;

    Key_matmul key_obj;

    int evaluate_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_MATMUL_EVALUATE_ITER",
                                  MATMUL_EVALUATE_ITER_V1);
    key_obj.Layout = Layout;
    key_obj.transpose_input = transpose_input;
    key_obj.transpose_filter = transpose_filter;
    key_obj.m = m;
    key_obj.k = k;
    key_obj.n = n;
    key_obj.alpha = alpha;
    key_obj.lda = lda;
    key_obj.ldb = ldb;
    key_obj.ldc = ldc;
    key_obj.thread_count = zenEnvObj.omp_num_threads;

    //First time 'if' condition fails for warm up.
    //Later batches get the path from the map
    //finds object in map
    auto found_obj = matmul_kernel_map.find(key_obj);

    if (found_obj != matmul_kernel_map.end() &&
            found_obj->second.first > evaluate_iteration) {
        //return the best kernel path
        zenEnvObj.zenGEMMalgo = matmul_kernel_map[key_obj].second;

        zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_filter, m, k,
                       n,
                       alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);
    }
    //Warm up step starts here.
    else {

        //Loop over all kernel definition
        for (int algo_type = 1; algo_type <= NUM_OF_ALGO ; algo_type++) {
            zenEnvObj.zenGEMMalgo = algo_type;

            //timer start
            gettimeofday(&start_n, 0);

            zenMatMul_gemm(zenEnvObj, true, Layout,transpose_input, transpose_filter, m, k,
                           n,
                           alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);
            //timer end
            gettimeofday(&end_n, 0);
            cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f +
                            (end_n.tv_usec - start_n.tv_usec) / 1000.0f; //time in milliseconds


            //update with minimum time taken
            if (cur_min_time < 0 || cur_algo_time < cur_min_time) {
                cur_min_time = cur_algo_time; //Minimum time
                best_algo = algo_type; //Algo with minimum time (1-4)
            }
        }

        //update value for given key after running all paths.
        if (found_obj != matmul_kernel_map.end()) {
            found_obj->second.first += 1;
            found_obj->second.second = best_algo;
        }
        //If element not in map then add new element
        else {
            matmul_kernel_map[key_obj] = {1,best_algo};
        }
    }

    zenEnvObj.zenGEMMalgo = best_algo;
    return zenEnvObj.zenGEMMalgo;
}

//Version 2
//For choosing best algo it runs all available algo
// makes use of tight FOR loop.
//SKIP iteration functionality added. During skip
// time is not calculated.
//Evaluation is done one time.
int auto_compute_matmul_v2(
    zendnn::zendnnEnv zenEnvObj,
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
    const int ldc
) {

    int best_algo; //storing path for best kernel (1-4)
    float cur_min_time = -1; //storing minimum time of kernel
    float cur_algo_time = 0; //current algorithm's execution time
    struct timeval start_n, end_n;

    Key_matmul key_obj;

    int skip_iteration = zendnn::zendnn_getenv_int("ZENDNN_MATMUL_SKIP_ITER",
                         MATMUL_SKIP_ITER_V2);

    key_obj.Layout = Layout;
    key_obj.transpose_input = transpose_input;
    key_obj.transpose_filter = transpose_filter;
    key_obj.m = m;
    key_obj.k = k;
    key_obj.n = n;
    key_obj.alpha = alpha;
    key_obj.lda = lda;
    key_obj.ldb = ldb;
    key_obj.ldc = ldc;
    key_obj.thread_count = zenEnvObj.omp_num_threads;

    //finds object in map
    auto found_obj = matmul_kernel_map.find(key_obj);

    //If Less than Skip iteration run default algo
    if (found_obj == matmul_kernel_map.end() ||
            found_obj->second.first < skip_iteration) {
        zenEnvObj.zenGEMMalgo = zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1;

        zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_filter, m, k,
                       n,
                       alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);

        if (found_obj != matmul_kernel_map.end()) {
            found_obj->second.first += 1;
        }
        else {
            //Iteration count 1
            matmul_kernel_map[key_obj] = {1,-1};
        }
    }
    //If Algo is set in map then get from the map
    else if (found_obj->second.second != -1) {
        //return the best kernel path
        zenEnvObj.zenGEMMalgo = matmul_kernel_map[key_obj].second;

        zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_filter, m, k,
                       n,
                       alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);
    }
    //Warm up step starts here.
    //If algo not set then Run this else
    else {

        //Loop over all kernel definition
        for (int algo_type = 1; algo_type <= NUM_OF_ALGO ; algo_type++) {
            zenEnvObj.zenGEMMalgo = algo_type;

            //timer start
            gettimeofday(&start_n, 0);


            zenMatMul_gemm(zenEnvObj, true, Layout,transpose_input, transpose_filter, m, k,
                           n,
                           alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);
            //timer end
            gettimeofday(&end_n, 0);
            cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f +
                            (end_n.tv_usec - start_n.tv_usec) / 1000.0f; //time in milliseconds

            //update with minimum time taken
            if (cur_min_time < 0 || cur_algo_time<cur_min_time) {
                cur_min_time = cur_algo_time; //Minimum time
                best_algo = algo_type; //Algo with minimum time (1-4)
            }
        }

        //update value for given key after running all paths.
        found_obj->second.second = best_algo;
        zenEnvObj.zenGEMMalgo = best_algo;
    }

    return zenEnvObj.zenGEMMalgo;
}


/*Verion 3
  key = Key_matmul
  value = < count, time, algo >

  count : Total number of times the unique layer ran.
  time : Best time for selected algorithm used in MAP Creation Phase
  algo : The Algo to use.
*/

//Runs when ZENDNN_GEMM_ALGO=0
//Makes use of graph_exe_count that is incremented by framework.
//Evaluates each algo during warmup(skip + evaluation) phase.
int auto_compute_matmul_v3(
    zendnn::zendnnEnv zenEnvObj,
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
    const int ldc
) {

    float cur_algo_time; //current algorithm's execution time
    struct timeval start_n, end_n;

    Key_matmul key_obj;

    //Number of iterations to run without creating map for each unique layer.
    int skip_iteration = zendnn::zendnn_getenv_int("ZENDNN_MATMUL_SKIP_ITER",
                         MATMUL_SKIP_ITER_V3);

    //Number of iterations to run for creating map for each layer.
    int evaluate_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_MATMUL_EVALUATE_ITER",
                                  MATMUL_EVALUATE_ITER_V3);

    key_obj.Layout = Layout;
    key_obj.transpose_input = transpose_input;
    key_obj.transpose_filter = transpose_filter;
    key_obj.m = m;
    key_obj.k = k;
    key_obj.n = n;
    key_obj.alpha = alpha;
    key_obj.lda = lda;
    key_obj.ldb = ldb;
    key_obj.ldc = ldc;
    key_obj.thread_count = zenEnvObj.omp_num_threads;

    //finds object in map
    auto found_obj = matmul_kernel_map2.find(key_obj);

    //If iteration count is less than Skip iteration then run default algo
    if (graph_exe_count < skip_iteration) {

        //Set algo 3 initially
        zenEnvObj.zenGEMMalgo = zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1;

        //If Key not found in map then time the algo and add new element to map
        if (found_obj == matmul_kernel_map2.end()) {

            //Time start
            gettimeofday(&start_n,0);

            zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_filter, m, k,
                           n,
                           alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);

            //Time end
            gettimeofday(&end_n,0);

            cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f +
                            (end_n.tv_usec - start_n.tv_usec) / 1000.0f; //time in milliseconds

            //Create new entry
            matmul_kernel_map2[key_obj] = {0, cur_algo_time, 3}; // {eval_count, time, algo}
        }
        else {
            zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_filter, m, k,
                           n,
                           alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);
        }
    }

    //Read Value from map.
    //Runs after skip iterations and evaluation iterations are done.
    else if (graph_exe_count >= evaluate_iteration + skip_iteration) {

        //Get best algo for given layer from MAP (tuple's 2nd index has algo)
        zenEnvObj.zenGEMMalgo = std::get<2>(matmul_kernel_map2[key_obj]);

        zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_filter, m, k,
                       n,
                       alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);
    }
    //Runs for evaluate iterations
    //Updates the map values accordingly
    else {

        //Get the number of iteration already ran and select Algo to run for current iteration
        //get<0>(found_obj->second) = count of iteration
        zenEnvObj.zenGEMMalgo = (std::get<0>(found_obj->second)%NUM_OF_ALGO) +1;
        std::get<0>(found_obj->second) += 1;

        //timer start
        gettimeofday(&start_n, 0);

        zenMatMul_gemm(zenEnvObj, true, Layout,transpose_input, transpose_filter, m, k,
                       n,
                       alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);
        //timer end
        gettimeofday(&end_n, 0);

        cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f +
                        (end_n.tv_usec - start_n.tv_usec) / 1000.0f; //time in milliseconds


        //If current run gives better timing then update
        if (cur_algo_time < std::get<1>(found_obj->second)) {
            std::get<1>(found_obj->second) = cur_algo_time; //Minimum time for chosen algo
            std::get<2>(found_obj->second) =
                zenEnvObj.zenGEMMalgo; //Algo with minimum time (1-4)
        }

    }

    return zenEnvObj.zenGEMMalgo;
}


/*verion 4
  Makes use of eval_count to decide when to fetch value from map.
  Used when framework doesn't increment the graph_exe_count.

  Map value tuple
  <
    iteration_count,
    time,
    algo
  >
*/
//Runs when ZENDNN_GEMM_ALGO=0
int auto_compute_matmul_v4(
    zendnn::zendnnEnv zenEnvObj,
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
    const int ldc
) {

    float cur_algo_time; //current algorithm's execution time
    struct timeval start_n, end_n;

    Key_matmul key_obj;

    //Number of iterations to run without creating map for each unique layer.
    int skip_iteration = zendnn::zendnn_getenv_int("ZENDNN_MATMUL_SKIP_ITER",
                         MATMUL_SKIP_ITER_V4);

    //Number of iterations to run for creating map for each layer.
    int evaluate_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_MATMUL_EVALUATE_ITER",
                                  MATMUL_EVALUATE_ITER_V4);

    key_obj.Layout = Layout;
    key_obj.transpose_input = transpose_input;
    key_obj.transpose_filter = transpose_filter;
    key_obj.m = m;
    key_obj.k = k;
    key_obj.n = n;
    key_obj.alpha = alpha;
    key_obj.lda = lda;
    key_obj.ldb = ldb;
    key_obj.ldc = ldc;
    key_obj.thread_count = zenEnvObj.omp_num_threads;

    //finds object in map
    auto found_obj = matmul_kernel_map2.find(key_obj);

    //If current iterations less than Skip iteration then run default algo.
    //Checks using the (0) element of map value that denotes count of iterations in map
    if (found_obj == matmul_kernel_map2.end() ||
            std::get<0>(found_obj->second) < skip_iteration) {

        //Set algo 3 initially
        zenEnvObj.zenGEMMalgo = zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1;

        //If Key not found in map then time the algo and add new element to map
        if (found_obj == matmul_kernel_map2.end()) {

            //Time start
            gettimeofday(&start_n,0);

            zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_filter, m, k,
                           n,
                           alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);

            //Time end
            gettimeofday(&end_n,0);

            cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f +
                            (end_n.tv_usec - start_n.tv_usec) / 1000.0f; //time in milliseconds

            //Create new entry
            matmul_kernel_map2[key_obj] = {1, cur_algo_time, 3}; // {iter_count, time, algo}
        }
        //If key found then increment the iter_count and run algo 3.
        else {
            std::get<0>(found_obj->second) += 1;
            zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_filter, m, k,
                           n,
                           alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);
        }
    }
    //Read Value from map.
    //Runs after skip iterations and evaluation iterations are done.
    else if (std::get<0>(found_obj->second) > evaluate_iteration + skip_iteration) {

        //Get best algo for given layer from MAP (In value tuple, 2nd index has path)
        zenEnvObj.zenGEMMalgo = std::get<2>(matmul_kernel_map2[key_obj]);

        zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_filter, m, k,
                       n,
                       alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);

    }
    //Updates the map values by running different algorithms
    else {

        //Get the number of iteration already ran and select Algo to run for current iteration
        //get<0>(found_obj->second) = count of iteration
        zenEnvObj.zenGEMMalgo = (std::get<0>(found_obj->second)%NUM_OF_ALGO) +1;
        std::get<0>(found_obj->second) += 1;
        //timer start
        gettimeofday(&start_n, 0);

        zenMatMul_gemm(zenEnvObj, true, Layout,transpose_input, transpose_filter, m, k,
                       n,
                       alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);
        //timer end
        gettimeofday(&end_n, 0);

        cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f +
                        (end_n.tv_usec - start_n.tv_usec) / 1000.0f; //time in milliseconds


        //If current run gives better timing then update
        if (cur_algo_time < std::get<1>(found_obj->second)) {
            std::get<1>(found_obj->second) = cur_algo_time; //Minimum time for chosen algo
            std::get<2>(found_obj->second) =
                zenEnvObj.zenGEMMalgo; //Algo with minimum time (1-4)
        }

    }

    return zenEnvObj.zenGEMMalgo;
}
