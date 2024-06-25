/*******************************************************************************
* Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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
*
*******************************************************************************/

/* Steps:
 *  1. create engine and stream
 *  2. create user memory (input, indices, offsets, weights)
 *  3. create memory descriptor
 *  4. create embedding_bag descriptor
 *  5. create embedding_bag primitive descriptor
 *  6. create embedding_bag primitive
 *  7. execute the embedding_bag primitive
 */

//IMP => export ZENDNN_VERBOSE=1
//ZENDNN_VERBOSE=1 bin/simple_conv_test cpu

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <fstream>
#include <numeric>
#include <string>
#include <math.h>
#include <cstdlib>
#include <string.h>
#include <cstring>

#ifndef _WIN32
#include <unistd.h>
#include <sys/time.h>
#endif

#include "quant_utils.hpp"
#include "other_utils.hpp"
#include "cmd_parser.hpp"
#include "zendnn_logging.hpp"
#include "zendnn_tpp.hpp"

#define   API_SUCCESS          (0)
#define   API_FAILURE          (1)

using namespace std;
using namespace zendnn;
using namespace zendnn::tpp;
using namespace zendnn::tpp::cpu;

using zen_bf16 = zendnn::tpp::bfloat16;

ZenTensor fp32_tensor(std::vector<int64_t> asize, engine aengine, float finit = 0.0) {

    auto weight = empty_tensor(asize, memory::data_type::f32, aengine);

    int64_t nelem = weight.numel();
    float* hndl   = weight.data_ptr<float>();

    for(auto i = 0; i < nelem ; ++i) {
	hndl[i] = finit;
    }

    return weight;
}

ZenTensor bf16_tensor(std::vector<int64_t> asize, engine aengine, float finit = 0.0) {


    auto weight = empty_tensor(asize, memory::data_type::bf16, aengine);

    int64_t   nelem  = weight.numel();
    zen_bf16* hndl   = weight.data_ptr<zen_bf16>();

    for(auto i = 0; i < nelem ; ++i) {
	hndl[i] = chfp32bf16(finit);
    }

    return weight;
}

ZenTensor fp32_linear_no_bias(engine aengine) {
    ZenTensor weights = fp32_tensor({320,80,32,16,2}, aengine, 1.0);
    ZenTensor x       = fp32_tensor({1, 32, 5120}, aengine, 1.0);

    auto sizes = x.sizes();
    auto wt_sizes = weights.sizes();
    sizes[2] = wt_sizes[0] * wt_sizes[3];
    auto t_out = x.new_empty(sizes);

    tpp_linear_nobias_forward_cpu(t_out, x, weights);
    return t_out;
}

ZenTensor bf16_linear_no_bias(engine aengine) {
    ZenTensor weights = bf16_tensor({320,80,32,16,2}, aengine, 1.0);
    ZenTensor x       = bf16_tensor({1, 32, 5120}, aengine, 1.0);

    auto sizes = x.sizes();
    auto wt_sizes = weights.sizes();
    sizes[2] = wt_sizes[0] * wt_sizes[3];
    auto t_out = x.new_empty(sizes);

    tpp_linear_nobias_forward_cpu(t_out, x, weights);
    return t_out;
}

int linear_no_bias_test(engine aengine) {

    int  status   =  API_SUCCESS;

    auto fp32_res = fp32_linear_no_bias(aengine);
    auto bf16_res = bf16_linear_no_bias(aengine);

    auto bf16_conv = zendnn::tpp::empty_like(fp32_res);

    if (chbf16fp32mem(bf16_conv, bf16_res) == QUANT_UTILS_FAILURE){
	zendnnVerbose(ZENDNN_TESTLOG, "tensor conversion(bf16->fp32) fails");
	status =  API_FAILURE;
    }
    else {
	auto count = diff_mem(fp32_res, bf16_conv, 1e-02);
	if (count){
	    zendnnVerbose(ZENDNN_TESTLOG, "tensor comparison fails at", count, " points");
	    status =  API_FAILURE;
	}
    }

    return status;
}

int main(){

    int status = API_SUCCESS;

    engine::kind engine_kind = engine::kind::cpu;
    engine eng(engine_kind, 0);
    zendnnVerbose(ZENDNN_TESTLOG, "cpu engine created");

    status = linear_no_bias_test(eng);

    if (status == API_SUCCESS)
      zendnnInfo(ZENDNN_TESTLOG,
                 "ZenDNN API test for tpp integration successful.");
    else
      zendnnInfo(ZENDNN_TESTLOG,
                 "ZenDNN API test for tpp integration fails.");

    return status;
}
