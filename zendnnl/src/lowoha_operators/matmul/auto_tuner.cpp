/*******************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

#include "lowoha_operators/matmul/auto_tuner.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {

/**
 * @brief Maps a toggle value to a specific matmul algorithm.
 *
 * This function is used by the autotuner to select between different matmul
 * algorithms during the evaluation phase. It converts a numeric toggle value
 * into the corresponding algorithm enum.
 *
 * @param toggle_ Integer value used to select the algorithm:
 *                - 1: Returns oneDNN blocked algorithm
 *                - 2: Returns AOCL BLIS blocked algorithm
 *
 * @return matmul_algo_t The selected algorithm based on the toggle value.
 *
 * @note This function is typically called with (iteration_count % NUM_OF_ALGO) + 1
 *       to cycle through available algorithms during autotuning.
 */
matmul_algo_t get_algo(int toggle_) {
  if (toggle_ == 1) {
    return matmul_algo_t::onednn_blocked;
  }
  else {
    return matmul_algo_t::aocl_blis_blocked;
  }
}

unsigned int get_auto_tuner_iter(std::string val, bool is_skip) {
  char *skip_env_var = std::getenv(val.c_str());
  if (skip_env_var) {
    return std::stoi(skip_env_var);
  }
  return is_skip ? MATMUL_SKIP_ITER : MATMUL_EVALUATE_ITER;
}

matmul_algo_t auto_compute_matmul_v1(char layout, char transA, char transB,
                                     int M,
                                     int N, int K, float alpha, const void *A, int lda, const void *B, int ldb,
                                     float beta, void *C, int ldc, data_types &dtypes,
                                     zendnnl::ops::matmul_algo_t kernel, char mem_format_a, char mem_format_b,
                                     lowoha_params &lowoha_param, batch_params_t &batch_params,
                                     const void *bias, bool is_weights_const) {

  //It is used to know the size of bin.
  // unsigned int autoBinSize = zendnn::zendnn_getenv_int("ZENDNNL_AUTO_BIN_SIZE", 1);
  // if (autoBinSize == 0) {
  //     autoBinSize = 1;
  // }
  //Simplified Map having Key as struct and value as Algo.
  static std::unordered_map<Key_matmul, matmul_algo_t> matmul_kernel_map;

  //Map value is tuple of (iteration count, execution time of algo, Algo Path)
  static std::unordered_map<Key_matmul,std::tuple<unsigned int, float, matmul_algo_t>>
      matmul_kernel_map1_helper;

  Key_matmul key_obj_auto(transA, transB, M, K,
                          N, lda, ldb, B, (int32_t)matmul_algo_t::none);

  double cur_algo_time = 0.0;
  key_obj_auto.weights = is_weights_const? B : nullptr;

  profiler_t profiler;

  static unsigned int skip_iter = get_auto_tuner_iter("ZENDNNL_MATMUL_SKIP_ITER",
                                  true);
  static unsigned int evaluate_iter =
    get_auto_tuner_iter("ZENDNNL_MATMUL_EVAL_ITER", false);
  //finds object in map
  auto found_obj = matmul_kernel_map1_helper.find(key_obj_auto);

  //If current iterations less than Skip iteration then run default algo.
  //Checks using the (0) element of map value that denotes count of iterations in map
  if (found_obj == matmul_kernel_map1_helper.end() ||
      std::get<0>(found_obj->second) < skip_iter) {

    apilog_info("AutoTuner SKIP Iteration");
    //Set onednn blocked initially
    kernel = matmul_algo_t::onednn_blocked;

    //If Key not found in map then time the algo and add new element to map
    if (found_obj == matmul_kernel_map1_helper.end()) {

      //Time start
      profiler.tbp_start();

      matmul_kernel_wrapper(layout, transA, transB,
                            M, N, K, alpha,
                            A, lda, B,
                            ldb, beta, C, ldc,
                            dtypes, kernel, mem_format_a, mem_format_b,
                            lowoha_param, batch_params, bias, is_weights_const, true);
      //Time end
      profiler.tbp_stop();
      cur_algo_time = profiler.tbp_elapsedtime();

      //Create new entry
      get_lowoha_mutex().lock();
      //Map value is tuple of (iteration count, execution time of algo, Algo Path)
      matmul_kernel_map1_helper[key_obj_auto] = {1, cur_algo_time, matmul_algo_t::onednn_blocked};
      //Simplified Map having Key as struct and value as Algo.
      matmul_kernel_map[key_obj_auto] = matmul_algo_t::onednn_blocked;
      get_lowoha_mutex().unlock();
    }
    //If key found then increment the iter_count and run next algo.
    else {
      kernel = get_algo((std::get<0>(found_obj->second)%NUM_OF_ALGO) +1);
      std::get<0>(found_obj->second) += 1;
      matmul_kernel_wrapper(layout, transA, transB,
                            M, N, K, alpha,
                            A, lda, B,
                            ldb, beta, C, ldc,
                            dtypes, kernel,
                            mem_format_a, mem_format_b, lowoha_param, batch_params,
                            bias, is_weights_const, true);
    }
  }
  //Read Value from map.
  //Runs after skip iterations and evaluation iterations are done.
  else if (std::get<0>(found_obj->second) == evaluate_iter +
           skip_iter) {
    //Get best algo for given layer from MAP
    kernel = matmul_kernel_map[key_obj_auto];

    matmul_kernel_wrapper(layout, transA, transB,
                          M, N, K, alpha,
                          A, lda, B,
                          ldb, beta, C, ldc,
                          dtypes, kernel,
                          mem_format_a, mem_format_b, lowoha_param, batch_params, bias, is_weights_const,
                          true);
  }
  //Updates the map values by running different algorithms
  else {

    //Get the number of iteration already ran and select Algo to run for current iteration
    //get<0>(found_obj->second) = count of iteration
    kernel = get_algo((std::get<0>(found_obj->second)%NUM_OF_ALGO) +1);
    std::get<0>(found_obj->second) += 1;
    //Time start
    profiler.tbp_start();

    matmul_kernel_wrapper(layout, transA, transB,
                          M, N, K, alpha,
                          A, lda, B,
                          ldb, beta, C, ldc,
                          dtypes, kernel,
                          mem_format_a, mem_format_b, lowoha_param, batch_params,
                          bias, is_weights_const, true);

    //Time end
    profiler.tbp_stop();
    cur_algo_time = profiler.tbp_elapsedtime();
    apilog_info("AutoTuner Evaluate Iteration algo:",
                (int32_t)kernel, " time:",cur_algo_time);
    //If current run gives better timing then update
    if (cur_algo_time < std::get<1>(found_obj->second)) {
      std::get<1>(found_obj->second) = cur_algo_time; //Minimum time for chosen algo
      std::get<2>(found_obj->second) =
        kernel; //Algo with minimum time (1-NUM_OF_ALGO)
      matmul_kernel_map[key_obj_auto] = kernel;
    }

  }
  return kernel;
}

matmul_algo_t auto_compute_matmul_v2(char layout, char transA, char transB,
                                     int M,
                                     int N, int K, float alpha, const void *A, int lda, const void *B, int ldb,
                                     float beta, void *C, int ldc, data_types &dtypes,
                                     zendnnl::ops::matmul_algo_t kernel, char mem_format_a, char mem_format_b,
                                     lowoha_params &lowoha_param, batch_params_t &batch_params,
                                     const void *bias, bool is_weights_const) {
  //Map value is toggle_ value for algo
  static std::unordered_map<Key_matmul,unsigned int>
  matmul_kernel_map1_helper;

  Key_matmul key_obj_auto(transA, transB, M, K,
                          N, lda, ldb, B, (int32_t)matmul_algo_t::none);

  double cur_algo_time = 0.0;
  key_obj_auto.weights = is_weights_const? B : nullptr;

  profiler_t profiler;

  static unsigned int skip_iter = get_auto_tuner_iter("ZENDNNL_MATMUL_SKIP_ITER",
                                  true);
  static unsigned int evaluate_iter =
    get_auto_tuner_iter("ZENDNNL_MATMUL_EVAL_ITER", false);
  static std::vector<double> algo_time_vec(skip_iter + evaluate_iter, 0.0);

  //finds object in map
  auto found_obj = matmul_kernel_map1_helper.find(key_obj_auto);

  //If current iterations less than Skip iteration then run default algo.
  //Checks using the (0) element of map value that denotes count of iterations in map
  if (found_obj == matmul_kernel_map1_helper.end() ||
      found_obj->second < skip_iter) {

    apilog_info("AutoTuner SKIP Iteration");
    //Set onednn blocked initially
    kernel = matmul_algo_t::onednn_blocked;
    //If Key not found in map then time the algo and add new element to map
    if (found_obj == matmul_kernel_map1_helper.end()) {

      //Time start
      profiler.tbp_start();

      matmul_kernel_wrapper(layout, transA, transB,
                            M, N, K, alpha,
                            A, lda, B,
                            ldb, beta, C, ldc,
                            dtypes, kernel, mem_format_a, mem_format_b,
                            lowoha_param, batch_params, bias, is_weights_const, true);
      //Time end
      profiler.tbp_stop();
      cur_algo_time = profiler.tbp_elapsedtime();

      //Create new entry
      get_lowoha_mutex().lock();
      //Map value is toggle_ value for algo
      matmul_kernel_map1_helper[key_obj_auto] = {1};
      algo_time_vec[0] = INT_MAX;
      get_lowoha_mutex().unlock();
    }
    //If key found then increment the iter_count and run next algo.
    else {
      unsigned int iter_num = found_obj->second;
      kernel = get_algo((iter_num % NUM_OF_ALGO) + 1);
      found_obj->second += 1;
      algo_time_vec[iter_num] = INT_MAX;
      matmul_kernel_wrapper(layout, transA, transB,
                            M, N, K, alpha,
                            A, lda, B,
                            ldb, beta, C, ldc,
                            dtypes, kernel,
                            mem_format_a, mem_format_b, lowoha_param, batch_params,
                            bias, is_weights_const, true);
    }
  }
  //Read Value from map.
  //Runs after skip iterations and evaluation iterations are done.
  else if (found_obj->second == evaluate_iter + skip_iter) {
    //Get best algo for given layer from MAP
    // kernel = matmul_kernel_map[key_obj_auto];
    static matmul_algo_t best_kernel = get_algo((std::distance(
                                         algo_time_vec.begin(),
                                         std::min_element(algo_time_vec.begin(), algo_time_vec.end())))%NUM_OF_ALGO + 1);

    kernel = best_kernel;
    matmul_kernel_wrapper(layout, transA, transB,
                          M, N, K, alpha,
                          A, lda, B,
                          ldb, beta, C, ldc,
                          dtypes, kernel,
                          mem_format_a, mem_format_b, lowoha_param, batch_params,
                          bias, is_weights_const, true);
  }
  //Updates the map values by running different algorithms
  else {
    //Get the number of iteration already ran and select Algo to run for current iteration
    //get<0>(found_obj->second) = count of iteration
    unsigned int iter_num = found_obj->second;
    kernel = get_algo((iter_num % NUM_OF_ALGO) + 1);
    found_obj->second += 1;
    //Time start
    profiler.tbp_start();

    matmul_kernel_wrapper(layout, transA, transB,
                          M, N, K, alpha,
                          A, lda, B,
                          ldb, beta, C, ldc,
                          dtypes, kernel,
                          mem_format_a, mem_format_b, lowoha_param, batch_params,
                          bias, is_weights_const, true);

    //Time end
    profiler.tbp_stop();
    cur_algo_time = profiler.tbp_elapsedtime();
    apilog_info("AutoTuner Evaluate Iteration algo:",
                (int32_t)kernel, " time:",cur_algo_time);

    algo_time_vec[iter_num] += cur_algo_time;
  }
  return kernel;
}

} // namespace lowoha
} // namespace zendnnl