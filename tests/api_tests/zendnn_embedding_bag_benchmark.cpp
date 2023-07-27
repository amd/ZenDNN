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
 *  1. create engin and stream
 *  2. create user memory (input, indices, offsets, weights)
 *  3. create memory descriptor
 *  4. create embedding_bag descriptor
 *  5. create embedding_bag primitive descriptor
 *  6. create embedding_bag primitive
 *  7. execute the embedding_bag primitive
 */

//IMP => export ZENDNN_VERBOSE=1
//ZENDNN_VERBOSE=1 bin/simple_conv_test cpu

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
#ifndef _WIN32
    #include <unistd.h>
    #include <sys/time.h>
#endif
#include "test_utils.hpp"
#include "zendnn_logging.hpp"

#define   API_SUCCESS          (0)
#define   API_FAILURE          (1)

using namespace std;
using namespace zendnn;

/* ground truth */
std::vector<float> exp_sum_nwt_npd_128{546.56,1058.56,1570.56,2082.56,2594.56};
std::vector<float> exp_sum_wt_npd_128{883.84,1651.84,2419.84,3187.84,3955.84};
std::vector<float> exp_sum_nwt_npd_64{232.32,488.32,744.32,1000.32,1256.32};
std::vector<float> exp_sum_wt_npd_64{380.48,764.48,1148.48,1532.48,1916.48};
std::vector<float> exp_sum_nwt_npd_10{30.9,70.9,110.9,150.9,190.9};
std::vector<float> exp_sum_wt_npd_10{51.35,111.35,171.35,231.35,291.35};
std::vector<float> exp_sum_nwt_pd_128{209.28,1058.56,1570.56,2082.56,2594.56};
std::vector<float> exp_sum_wt_pd_128{209.28,1651.84,2419.84,3187.84,3955.84};
std::vector<float> exp_sum_nwt_pd_64{84.16,488.32,744.32,1000.32,1256.32};
std::vector<float> exp_sum_wt_pd_64{84.16,764.48,1148.48,1532.48,1916.48};
std::vector<float> exp_sum_nwt_pd_10{10.45,70.9,110.9,150.9,190.9};
std::vector<float> exp_sum_wt_pd_10{10.45,111.35,171.35,231.35,291.35};
std::vector<float> exp_mean_npd_128{273.28,529.28,785.28,1041.28,1297.28};
std::vector<float> exp_max_npd_128{337.28,593.28,849.28,1105.28,1361.28};


/* functor for float comparison */
class compare_float_t {
  public:
    compare_float_t(float tol = 1e-05):tolerance{tol} {}

    bool operator()(const float a, const float b) {
        if (isnan(a) || isnan(b)) {
            return false;
        }
        if ((a -b) > tolerance) {
            return false;
        }
        if ((b -a) > tolerance) {
            return false;
        }

        return true;
    }

  private:
    float tolerance;
};

/* compare two vectors */
bool compare_vectors(const std::vector<float> &a, const std::vector<float> &b) {

    compare_float_t cmp_op{1e-02};

    if (a.size() != b.size()) {
        return false;
    }

    for (auto i = 0; i < a.size(); ++i) {
        if (!cmp_op(a[i],b[i])) {
            return false;
        }
    }

    return true;
};

/* display a vector */
std::string vec_str(std::string vd, const std::vector<float> &a) {

    auto sz = a.size();

    std::string out = vd + "{";
    for (auto i = 0; i < sz -1; ++i) {
        out = out + std::to_string(a[i]) + ";";
    }

    out = out + std::to_string(a[sz -1]) + "}";

    return out;
};

/* create embedding table */
memory create_embedding_table(engine eng, uint32_t rows, uint32_t width) {

    const float incr = 0.01;

    auto table = memory({{rows, width},
        memory::data_type::f32,
        memory::format_tag::ab}, eng);

    float *hndl = (float *)table.get_data_handle();

    for (auto i = 0; i < rows ; ++i) {
        for (auto j = 0; j < width; ++j) {
            hndl[j + i*width] = i + 1.0 + j*incr;
        }
    }

    return table;
}

/* create indices */
memory create_indices(engine eng, uint32_t len) {

    auto indices = memory({{len},
        memory::data_type::s32,
        memory::format_tag::a}, eng);

    int32_t *hndl = (int32_t *)indices.get_data_handle();

    /* indices should be in the range of 0 to rows */
    for (auto i = 0; i < len; ++i) {
        hndl[i] = i;
    }

    return indices;
}

/* create offsets */
memory create_offsets(engine eng, uint32_t len) {

    const uint32_t incr = 2;
    uint32_t       bags = len/incr;

    auto offsets = memory({{bags},
        memory::data_type::s32,
        memory::format_tag::a}, eng);

    int32_t *hndl = (int32_t *)offsets.get_data_handle();

    /* first offset should always be zero */
    uint32_t offset = 0;
    for (auto i = 0; i < bags; ++i) {
        hndl[i] = offset;
        offset += incr;
    }

    return offsets;
}

/* create weights */
memory create_weights(engine eng, uint32_t len) {

    const uint32_t modulo = 2;

    auto weights = memory({{len},
        memory::data_type::f32,
        memory::format_tag::a}, eng);

    float *hndl = (float *)weights.get_data_handle();

    for (auto i = 0; i < len; ++i) {
        hndl[i]  = 1.0 + float(i % modulo);
    }

    return weights;

}

/* create output */
memory create_output(engine eng, uint32_t rows, uint32_t width) {

    /* create destination buffer */
    auto dst = memory({{rows, width},
        memory::data_type::f32,
        memory::format_tag::ab}, eng);

    return dst;
}

/* sum the bags for verification */
std::vector<float> sum_bags(memory &dst) {
    std::vector<float> sum;

    auto dst_dims       = dst.get_desc().dims();
    auto num_bags       = dst_dims[0];
    auto dim_embedding  = dst_dims[1];

    float *hndl         = (float *)dst.get_data_handle();

    int32_t idx_base    = 0;
    for (auto bag = 0; bag < num_bags; ++bag) {
        float bag_sum = 0.0;
        for (auto j = 0; j < dim_embedding; ++j) {
            bag_sum += hndl[idx_base +j];
        }

        sum.push_back(bag_sum);
        idx_base += dim_embedding;
    }

    return sum;
}

/* execute embedding_bag */
void exec_embedding_bag(engine eng, stream s, memory table,
                        memory indices, memory offsets, memory weights,
                        memory bags, algorithm alg, uint32_t num_threads,
                        bool is_weights, int32_t padding_idx) {

    auto table_md   = memory::desc(table.get_desc());
    auto indices_md = memory::desc(indices.get_desc());
    auto offsets_md = memory::desc(offsets.get_desc());
    auto weights_md = memory::desc(weights.get_desc());
    auto bags_md    = memory::desc(bags.get_desc());

    auto emdb_d = embedding_bag::desc();

    if (!is_weights) {
        if (padding_idx < 0)
            emdb_d = embedding_bag::desc(prop_kind::forward_inference,
                                         alg,
                                         num_threads,
                                         table_md,
                                         indices_md,
                                         offsets_md,
                                         bags_md);
        else
            emdb_d = embedding_bag::desc(prop_kind::forward_inference,
                                         alg,
                                         num_threads,
                                         table_md,
                                         indices_md,
                                         offsets_md,
                                         bags_md,
                                         padding_idx);
    }
    else {
        if (padding_idx < 0)
            emdb_d = embedding_bag::desc(prop_kind::forward_inference,
                                         alg,
                                         num_threads,
                                         table_md,
                                         indices_md,
                                         offsets_md,
                                         weights_md,
                                         bags_md);
        else
            emdb_d = embedding_bag::desc(prop_kind::forward_inference,
                                         alg,
                                         num_threads,
                                         table_md,
                                         indices_md,
                                         offsets_md,
                                         weights_md,
                                         bags_md,
                                         padding_idx);
    }

    auto emdb_pd = embedding_bag::primitive_desc(emdb_d, eng);
    auto emdb    = embedding_bag(emdb_pd);

    if (!is_weights) {
        emdb.execute(s, {{ZENDNN_ARG_SRC_0, table},
            {ZENDNN_ARG_SRC_1, indices},
            {ZENDNN_ARG_SRC_2, offsets},
            {ZENDNN_ARG_DST, bags}
        });
    }
    else {
        emdb.execute(s, {{ZENDNN_ARG_SRC_0, table},
            {ZENDNN_ARG_SRC_1, indices},
            {ZENDNN_ARG_SRC_2, offsets},
            {ZENDNN_ARG_SRC_3, weights},
            {ZENDNN_ARG_DST, bags}
        });
    }
}

bool run_test(std::string test_id, engine eng, stream s, uint32_t width,
              algorithm alg, bool is_wt, int32_t padidx,
              std::vector<float> &expected, uint32_t emb_rows_in, uint32_t emb_nthr_in,
              uint32_t num_iteration_in) {

    //parameters
    uint32_t emb_rows  = emb_rows_in;
    uint32_t emb_nthr  = emb_nthr_in;
    uint32_t num_iteration = num_iteration_in;
    /* create embedding table */
    memory table = create_embedding_table(eng, emb_rows, width);

    /* create indices */
    memory indices = create_indices(eng, emb_rows);

    /* create offsets */
    memory offsets = create_offsets(eng, emb_rows);

    /* create weights */
    memory weights = create_weights(eng, emb_rows);

    /* create output */
    auto   bag_count      = offsets.get_desc().dims()[0];
    memory bags           = create_output(eng, bag_count, width);

    /* execute emb */
    for (int i = 0; i < num_iteration; ++i) {
        exec_embedding_bag(eng, s, table, indices, offsets, weights, bags,
                           alg, emb_nthr, is_wt, padidx);
    }
    /* compare against ground truth */
    auto actual = sum_bags(bags);

    if (!compare_vectors(expected, actual)) {
        zendnnError(ZENDNN_TESTLOG, test_id, ":", vec_str("exp", expected),":",
                    vec_str("act", actual));

        return false;
    }

    return true;
}

int main(int argc, char **argv) {

    int status = API_SUCCESS;

    zendnnInfo(ZENDNN_TESTLOG, "ZenDNN API test for embedding_bag starts");

    uint32_t emb_rows  = atoi(argv[2]);
    uint32_t emb_nthr  = atoi(argv[3]);
    uint32_t num_iteration = atoi(argv[4]) ;


    /* create engine kind */
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    engine eng(engine_kind, 0);
    zendnnVerbose(ZENDNN_TESTLOG, "cpu engine created");

    /* create stream */
    stream s(eng);
    zendnnVerbose(ZENDNN_TESTLOG, "stream created");

    /* run tests */
    if (!run_test("sm_nw_np_128", eng, s, 128, algorithm::embedding_bag_sum,
                  false, -1, exp_sum_nwt_npd_128, emb_rows, emb_nthr, num_iteration)) {
        status = API_FAILURE;
    }

    if (!run_test("sm_w_np_128", eng, s, 128, algorithm::embedding_bag_sum,
                  true, -1, exp_sum_wt_npd_128, emb_rows, emb_nthr, num_iteration)) {
        status = API_FAILURE;
    }

    if (!run_test("sm_nw_np_64", eng, s, 64, algorithm::embedding_bag_sum,
                  false, -1, exp_sum_nwt_npd_64, emb_rows, emb_nthr, num_iteration)) {
        status = API_FAILURE;
    }

    if (!run_test("sm_w_np_64", eng, s, 64, algorithm::embedding_bag_sum,
                  true, -1, exp_sum_wt_npd_64, emb_rows, emb_nthr, num_iteration)) {
        status = API_FAILURE;
    }

    if (!run_test("sm_nw_np_10", eng, s, 10, algorithm::embedding_bag_sum,
                  false, -1, exp_sum_nwt_npd_10, emb_rows, emb_nthr, num_iteration)) {
        status = API_FAILURE;
    }

    if (!run_test("sm_w_np_10", eng, s, 10, algorithm::embedding_bag_sum,
                  true, -1, exp_sum_wt_npd_10, emb_rows, emb_nthr, num_iteration)) {
        status = API_FAILURE;
    }

    if (!run_test("mn_np_128", eng, s, 128, algorithm::embedding_bag_mean,
                  false, -1, exp_mean_npd_128, emb_rows, emb_nthr, num_iteration)) {
        status = API_FAILURE;
    }

    if (!run_test("mx_w_np_128", eng, s, 128, algorithm::embedding_bag_max,
                  false, -1, exp_max_npd_128, emb_rows, emb_nthr, num_iteration)) {
        status = API_FAILURE;
    }


    if (status == API_SUCCESS)
        zendnnInfo(ZENDNN_TESTLOG,
                   "ZenDNN API test for embedding_bag successful.");
    else
        zendnnInfo(ZENDNN_TESTLOG,
                   "ZenDNN API test for embedding_bag fails.");

    return status;
}
