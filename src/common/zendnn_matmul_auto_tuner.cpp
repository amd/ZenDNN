/*******************************************************************************
* Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "zendnn_private.hpp"
#include <time.h>
#include <sstream>
#include <fstream>
#include "zendnn_logging.hpp"
#include "utils.hpp"

using namespace zendnn;

//Total num of algo available
#define NUM_OF_ALGO 2
#define NUM_OF_ALGO_INT8 2
//Total num of struct members and algo field in MAP.
#define NUM_MAP_VALUES 10
//CPU information size
#define CPU_INFO_SIZE 12

//Skip Iterations for auto tuner
//Can be set by environment variable ZENDNN_MATMUL_SKIP_ITER
#define MATMUL_SKIP_ITER_INT8 2
#define MATMUL_SKIP_ITER_FP32 2
//Evaluate iterations for auto tuner
//Can be set by environment variable ZENDNN_MATMUL_EVALUATE_ITER
#define MATMUL_EVALUATE_ITER_INT8 2
#define MATMUL_EVALUATE_ITER_FP32 2

//This tracks the no. of times graph executed
// from the framework.
int graph_exe_count = -1;

//enum defines the persistent map controls
//0: disables
//1: Write the map on the file
//2: Read the map from file
enum persistentMapType {
    DISABLE = 0,
    WRITE = 1,
    READ = 2,
};

//Simplified Map having Key as struct and value as Algo.
std::unordered_map<Key_matmul, unsigned int>
matmul_kernel_map;

//Map value is tuple of (iteration count, execution time of algo, Algo Path)
std::unordered_map<Key_matmul,std::tuple<unsigned int, float, unsigned int>>
        matmul_kernel_map1_helper;

//Map value is tuple of (vector<pair>, execution time of algo, Algo Path)
//Each element of vector represents pair of
//iteration count and average time for each algo(iteration count, average time)
std::unordered_map<Key_matmul,std::tuple<std::vector<std::pair<unsigned int,float>>, float, unsigned int>>
        matmul_kernel_map2_helper;

//Writing in the file from the map.
int map_write_to_file() {

    //Fetch File name given by user.
    char *fname = getenv("ZENDNN_MATMUL_MAP_FILE");
    if (fname == NULL) {
        fname = (char *)"key_matmul_map.csv";
    }

    std::ofstream file;
    Key_matmul sobj;
    file.open(fname,std::ios::out);

    //File not open
    if (!file.is_open()) {
        return 1;
    }

    //Put Main header.
    file<<"ZENDNN MatMul Primitive Map for selecting best Algo path\n";
    int cpu_i[CPU_INFO_SIZE];

    //get the cpuid in cpu_i
    if (getCpuID_brandString(cpu_i)==0) {
        file<<(char *)cpu_i<<"\n";
    }
    else {
        file<<"Architecture brand string not found\n";
    }
    file<<"Transpose Input,Transpose Filter,M,K,N,lda,ldb,ldc,Thread,Algo\n";
    for (auto itr=matmul_kernel_map.begin(); itr!=matmul_kernel_map.end(); itr++) {
        sobj = (*itr).first;
        unsigned int algo_type = (*itr).second;

        file<<sobj.transpose_input<<","<<sobj.transpose_weights<<","<<sobj.m<<","<<sobj.k<<","<<sobj.n<<","<<sobj.lda<<","<<sobj.ldb<<","<<sobj.ldc<<","<<sobj.thread_count<<","<<algo_type<<"\n";
    }

    file.close();

    zendnnInfo(ZENDNN_ALGOLOG, "MAP FILE LOCATION ", fname);

    return 0;
}


//Reading already existing map
int map_read_from_file() {

    //Fetch File name given by user.
    char *fname = getenv("ZENDNN_MATMUL_MAP_FILE");

    //File not specified
    if (fname == NULL) {
        return 1;
    }

    Key_matmul obj;
    std::vector<unsigned int> map_data; //Store each comma separated value
    std::string temp1,temp2;
    const char *cpu_str; //To store cpu name as char array.
    int *temp_cpu_name;

    std::ifstream file;
    file.open(fname,std::ios::in);

    int cpu_i[CPU_INFO_SIZE];
    if (getCpuID_brandString(cpu_i)!=0) {
        zendnnError(ZENDNN_ALGOLOG, "Could not fetch CPUID.");
        return 1;
    }

    if (file.fail()) {
        return 1;
    }
    else {
        //read first header
        getline(file,temp1);

        //Read CPU Name and Check
        getline(file,temp1);

        cpu_str = temp1.c_str();
        temp_cpu_name = (int *)cpu_str;

        //Check if CPU name in File is same as current CPU.
        for (int i=0; i<CPU_INFO_SIZE; i++) {
            if (temp_cpu_name[i]!=cpu_i[i]) {
                return 1;
            }
        }

        //Read Third line of header
        getline(file,temp1);

        //Read values from file
        while (getline(file,temp1)) {
            std::stringstream line(temp1);

            //Checking for invalid value provided in the file.
            try {
                //Retrieve each value from the line(comma separated).
                while (getline(line, temp2, ',')) {
                    map_data.push_back(stod(temp2));
                }

                //Few or more number of values in line
                if (map_data.size() != NUM_MAP_VALUES) {
                    return 1;
                }

            }
            catch (std::invalid_argument const &e) {
                return 1;
            }

            obj.transpose_input = map_data[0];
            obj.transpose_weights = map_data[1];
            obj.m = map_data[2];
            obj.k = map_data[3];
            obj.n = map_data[4];
            obj.lda = map_data[5];
            obj.ldb = map_data[6];
            obj.ldc = map_data[7];
            obj.thread_count = map_data[8];

            // weight address is set to NULL for Persistent map feature.
            obj.weights = NULL;

            //Fill value in the map.
            matmul_kernel_map[obj] = map_data[9];

            map_data.clear();
        }
        file.close();
    }

    return 0;
}

/*INT8 AutoTuner
 *
 Based on iteration count
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
//Start
int auto_compute_matmul_int8(
    const impl::exec_ctx_t &ctx,
    zendnn::zendnnEnv zenEnvObj,
    int src_type,
    int dst_type,
    int bias_type,
    const bool Layout,
    const bool transpose_input,
    const bool transpose_weights,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const char *input,
    const int lda,
    const int8_t *weights,
    const int ldb,
    const char *bias,
    const impl::post_ops_t &po_ops,
    const float beta,
    char *dst,
    const int ldc,
    const int32_t zero_point_src,
    const int32_t zero_point_wei,
    const int32_t zero_point_dst,
    float do_sum,
    bool is_weights_const,
    float *src_scale,
    int src_scale_size,
    float *wei_scale,
    int wei_scale_size,
    float *dst_scales,
    int dst_scale_size,
    bool default_dst_scales,
    int scale_type
) {
    //It is used to know if weights address should be enabled or not for map.
    //0: disable, 1: enable.
    unsigned int mapType = zendnn::zendnn_getenv_int("ZENDNN_GEMM_MAP_TYPE",1);

    Key_matmul key_obj(transpose_input, transpose_weights, m, k, n, lda, ldb, ldc,
                       weights, zenEnvObj.omp_num_threads, false);

    //This condition makes sure that address
    //doesn't gets saved while using persistent map.
    key_obj.weights = mapType == 1 ? (int8_t *)weights : NULL;

    float cur_algo_time; //current algorithm's execution time
    struct timeval start_n, end_n;

    //Number of iterations to run without creating map for each unique layer.
    unsigned int skip_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_MATMUL_SKIP_ITER",
                                  MATMUL_SKIP_ITER_INT8);

    //Number of iterations to run for creating map for each layer.
    unsigned int evaluate_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_MATMUL_EVALUATE_ITER",
                                  MATMUL_EVALUATE_ITER_INT8);

    //finds object in map
    auto found_obj = matmul_kernel_map1_helper.find(key_obj);

    //If current iterations less than Skip iteration then run default algo.
    //Checks using the (0) element of map value that denotes count of iterations in map
    if (found_obj == matmul_kernel_map1_helper.end() ||
            std::get<0>(found_obj->second) < skip_iteration) {

        //Set AOCL GEMM algo for skip iterations.
        zenEnvObj.zenINT8GEMMalgo = zenINT8MatMulAlgoType::MATMUL_BLOCKED_AOCL_INT8;

        //If Key not found in map then time the algo and add new element to map
        if (found_obj == matmul_kernel_map1_helper.end()) {

            //Time start
#ifdef _WIN32
            auto start_n = std::chrono::high_resolution_clock::now();
#else
            gettimeofday(&start_n, 0);
#endif

            matmul_int8_wrapper(ctx, zenEnvObj, src_type, dst_type, bias_type, Layout,
                                transpose_input, transpose_weights,
                                m, k, n, alpha, input, lda, weights, ldb, bias, po_ops,
                                beta, (char *)dst, ldc, zero_point_src,
                                zero_point_wei, zero_point_dst, do_sum, is_weights_const, 
                                src_scale, src_scale_size, wei_scale, wei_scale_size,
                                dst_scales, dst_scale_size, default_dst_scales, scale_type);
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
            matmul_kernel_map1_helper[key_obj] = {1, cur_algo_time, zenINT8MatMulAlgoType::MATMUL_BLOCKED_AOCL_INT8}; // {iter_count, time, algo}
            matmul_kernel_map[key_obj] = zenINT8MatMulAlgoType::MATMUL_BLOCKED_AOCL_INT8;
        }
        //If key found then increment the iter_count and run next algo.
        else {
            zenEnvObj.zenINT8GEMMalgo = (std::get<0>(found_obj->second)%NUM_OF_ALGO_INT8)+1;
            std::get<0>(found_obj->second) += 1;
            matmul_int8_wrapper(ctx, zenEnvObj, src_type, dst_type, bias_type, Layout,
                                transpose_input, transpose_weights,
                                m, k, n, alpha, input, lda, weights, ldb, bias, po_ops,
                                beta, (char *)dst, ldc, zero_point_src,
                                zero_point_wei, zero_point_dst, do_sum, is_weights_const, 
                                src_scale, src_scale_size, wei_scale, wei_scale_size,
                                dst_scales, dst_scale_size, default_dst_scales, scale_type);
        }
    }
    //Read Value from map.
    //Runs after skip iterations and evaluation iterations are done.
    else if (std::get<0>(found_obj->second) > evaluate_iteration + skip_iteration) {

        //Get best algo for given layer from MAP
        zenEnvObj.zenINT8GEMMalgo = matmul_kernel_map[key_obj];
        matmul_int8_wrapper(ctx, zenEnvObj, src_type, dst_type, bias_type, Layout,
                            transpose_input, transpose_weights,
                            m, k, n, alpha, input, lda, weights, ldb, bias, po_ops,
                            beta, (char *)dst, ldc, zero_point_src,
                            zero_point_wei, zero_point_dst, do_sum, is_weights_const, 
                            src_scale, src_scale_size, wei_scale, wei_scale_size,
                            dst_scales, dst_scale_size, default_dst_scales, scale_type);
    }
    //Updates the map values by running different algorithms
    else {

        //Get the number of iteration already ran and select Algo to run for current iteration
        //get<0>(found_obj->second) = count of iteration
        zenEnvObj.zenINT8GEMMalgo = (std::get<0>(found_obj->second)%NUM_OF_ALGO_INT8)
                                    +1;
        std::get<0>(found_obj->second) += 1;
        //timer start
#ifdef _WIN32
        auto start_n = std::chrono::high_resolution_clock::now();
#else
        gettimeofday(&start_n, 0);
#endif
        matmul_int8_wrapper(ctx, zenEnvObj, src_type, dst_type, bias_type, Layout,
                            transpose_input, transpose_weights,
                            m, k, n, alpha, input, lda, weights, ldb, bias, po_ops,
                            beta, (char *)dst, ldc, zero_point_src,
                            zero_point_wei, zero_point_dst, do_sum, is_weights_const, 
                            src_scale, src_scale_size, wei_scale, wei_scale_size,
                            dst_scales, dst_scale_size, default_dst_scales, scale_type);
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
                zenEnvObj.zenINT8GEMMalgo; //Algo with minimum time (1-NUM_OF_ALGO)
            matmul_kernel_map[key_obj] = zenEnvObj.zenINT8GEMMalgo;
        }

    }

    return zenEnvObj.zenINT8GEMMalgo;
}
//END

/*FP32 AutoTuner
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
int auto_compute_matmul_fp32(
    zendnn::zendnnEnv zenEnvObj,
    std::pair<unsigned int, unsigned int> *persistent_map_flag,
    Key_matmul key_obj,
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
    const int ldc,
    bool is_weights_const
) {

    float cur_algo_time; //current algorithm's execution time
    struct timeval start_n, end_n;

    //Number of iterations to run without creating map for each unique layer.
    unsigned int skip_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_MATMUL_SKIP_ITER",
                                  MATMUL_SKIP_ITER_FP32);

    //Number of iterations to run for creating map for each layer.
    unsigned int evaluate_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_MATMUL_EVALUATE_ITER",
                                  MATMUL_EVALUATE_ITER_FP32);

    //finds object in map
    auto found_obj = matmul_kernel_map1_helper.find(key_obj);

    //If current iterations less than Skip iteration then run default algo.
    //Checks using the (0) element of map value that denotes count of iterations in map
    if (found_obj == matmul_kernel_map1_helper.end() ||
            std::get<0>(found_obj->second) < skip_iteration) {

        //Set algo 3 initially
        zenEnvObj.zenGEMMalgo = zenMatMulAlgoType::MATMUL_BLOCKED_JIT_FP32;

        //If Key not found in map then time the algo and add new element to map
        if (found_obj == matmul_kernel_map1_helper.end()) {

            //Time start
#ifdef _WIN32
            auto start_n = std::chrono::high_resolution_clock::now();
#else
            gettimeofday(&start_n, 0);
#endif

            zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_weights, m,
                           k, n, alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc,
                           is_weights_const);

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
            matmul_kernel_map1_helper[key_obj] = {1, cur_algo_time, zenMatMulAlgoType::MATMUL_BLOCKED_JIT_FP32}; // {iter_count, time, algo}
            matmul_kernel_map[key_obj] = zenMatMulAlgoType::MATMUL_BLOCKED_JIT_FP32;
        }
        //If key found then increment the iter_count and run next algo.
        else {
            zenEnvObj.zenGEMMalgo = (std::get<0>(found_obj->second)%NUM_OF_ALGO) +1;
            std::get<0>(found_obj->second) += 1;
            zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_weights, m,
                           k, n, alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc,
                           is_weights_const);
        }
    }
    //Read Value from map.
    //Runs after skip iterations and evaluation iterations are done.
    else if (std::get<0>(found_obj->second) > evaluate_iteration + skip_iteration) {

        //Get best algo for given layer from MAP
        zenEnvObj.zenGEMMalgo = matmul_kernel_map[key_obj];
        zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_weights, m,
                       k, n, alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc,
                       is_weights_const);

        //Writing Map in file.
        if (persistent_map_flag->first == persistentMapType::WRITE &&
                persistent_map_flag->second) {
            if (map_write_to_file()) {
                zendnnError(ZENDNN_ALGOLOG,
                            "Error occured while writing Persistent Map File. Check the file");
            }
            persistent_map_flag->second = 0;
        }

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
                       alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc,
                       is_weights_const);
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
                zenEnvObj.zenGEMMalgo; //Algo with minimum time (1-NUM_OF_ALGO)
            matmul_kernel_map[key_obj] = zenEnvObj.zenGEMMalgo;

            //To update the map file if any update occurs after writing map once.
            persistent_map_flag->second = 1;
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
    const int ldc,
    bool is_weights_const
) {
    unsigned int algo_type;

    Key_matmul key_obj(transpose_input, transpose_weights, m, k, n, lda, ldb, ldc,
                       weights, zenEnvObj.omp_num_threads, false);

    //Persistent Map
    //{ 0: disable, 1: write, 2:read }
    unsigned int persistent_map =
        zendnn::zendnn_getenv_int("ZENDNN_MATMUL_PERSISTENT_MAP",
                                  persistentMapType::DISABLE);

    //If Persistent map value not 0/1/2 then assume it as Disabled.
    if (persistent_map >2) {
        persistent_map = persistentMapType::DISABLE;
    }

    //It is used to know if weights address should be enabled or not for map.
    //0: disable, 1: enable.
    unsigned int mapType = zendnn::zendnn_getenv_int("ZENDNN_GEMM_MAP_TYPE",1);

    //These flags are passed in different versions to ensure that writing of map is done
    //only when needed.
    static std::pair<unsigned int, unsigned int> persistent_map_flag = {persistent_map, 1};

    //This condition makes sure that address
    //doesn't gets saved while using persistent map.
    key_obj.weights = mapType == 1 &&
                      persistent_map == persistentMapType::DISABLE ? weights : NULL;

    //Read operation from File (Persistent Map)
    if (persistent_map == persistentMapType::READ) {
        //Check if we need to read the map from file.
        if (persistent_map_flag.second) {
            if (map_read_from_file()) {

                zendnnError(ZENDNN_ALGOLOG,
                            "Persistent Map File Not Found or invalid value in file. Persistent feature won't work. Set ZENDNN_MATMUL_MAP_FILE environment variable. Executing with default algo.");
            }
            persistent_map_flag.second = 0;
        }
        //Check if given key exist in map or not
        if (matmul_kernel_map.find(key_obj) != matmul_kernel_map.end()) {
            zenEnvObj.zenGEMMalgo = matmul_kernel_map[key_obj];

            zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_weights, m,
                           k,
                           n,
                           alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc,
                           is_weights_const);

        }
        else {

            zenEnvObj.zenGEMMalgo = zenMatMulAlgoType::MATMUL_BLOCKED_JIT_FP32;

            zenMatMul_gemm(zenEnvObj, true, Layout, transpose_input, transpose_weights, m,
                           k,
                           n,
                           alpha, input, lda, weights, ldb, bias, relu, gelu, beta, output, ldc,
                           is_weights_const);

        }

        return zenEnvObj.zenGEMMalgo;
    }

    algo_type = auto_compute_matmul_fp32(zenEnvObj, &persistent_map_flag, key_obj,
                                         Layout, transpose_input, transpose_weights, m, k, n, alpha, input, lda, weights,
                                         ldb, bias, relu, gelu, beta, output, ldc, is_weights_const);
    return algo_type;
}
