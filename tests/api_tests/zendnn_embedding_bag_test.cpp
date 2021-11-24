/*******************************************************************************
* Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
#include <sys/time.h>
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
#include <unistd.h>
#include <string.h>

#include "test_utils.hpp"
#include "zendnn_logging.hpp"

#define   API_SUCCESS          (0)
#define   API_FAILURE          (1)

using namespace std;
using namespace zendnn;

/* parameters */
struct emb_params {
    int32_t   num_embedding = 10;
    int32_t   dim_embedding = 20;
    int32_t   num_indices   = 12;
    int32_t   num_bags      = 3;
    int32_t   padding_idx   = 1;
    int32_t   indices[12]   = {0,1,5, 1,4,2,6, 1,3,9,8,7};
    int32_t   offsets[3]    = {0,3,7};
    float     weights[12]   = {1,2,3,4,5,6,7,8,9,10,11,12};
};

/* expected otput */
float expected_output_sum_wt_pd[]    = {6840, 33380, 124420};
float expected_output_mean_nwt_npd[] = {1010, 1510, 2450};
float expected_output_max_nwt_pd[]   = {2210, 2610, 3810};

/* functor for float comparison */
class compare_float_t {
public:
    compare_float_t(float tol = 1e-05):tolerance{tol} {}

    bool operator()(float a, float b) {
        if(isnan(a) || isnan(b))
            return false;
        if((a -b) > tolerance)
            return false;
        if((b -a) > tolerance)
            return false;

        return true;
    }

private:
    float tolerance;
};

/* create embedding table */
memory create_embedding_table(engine eng, emb_params &params) {

    int32_t  &num_embedding = params.num_embedding;
    int32_t  &dim_embedding = params.dim_embedding;

    auto table = memory({{num_embedding, dim_embedding},
        memory::data_type::f32,
        memory::format_tag::ab}, eng);

    float *hndl = (float *)table.get_data_handle();

    for(auto i = 0; i < num_embedding*dim_embedding; ++i) {
        hndl[i] = float(i+1);
    }

    return table;
}

/* create indices */
memory create_indices(engine eng, emb_params &params) {

    int32_t &num_indices = params.num_indices;

    auto indices = memory({{num_indices},
        memory::data_type::s32,
        memory::format_tag::a}, eng);

    int32_t *hndl = (int32_t *)indices.get_data_handle();

    /* indices should be in the range of 0 to num_embedding */
    for(auto i = 0; i < num_indices; ++i) {
        hndl[i] = params.indices[i];
    }

    return indices;
}

/* create offsets */
memory create_offsets(engine eng, emb_params &params) {

    int32_t &num_bags = params.num_bags;

    auto offsets = memory({{num_bags},
        memory::data_type::s32,
        memory::format_tag::a}, eng);

    int32_t *hndl = (int32_t *)offsets.get_data_handle();

    /* first offset should always be zero */
    for(auto i = 0; i < num_bags; ++i) {
        hndl[i] = params.offsets[i];
    }

    return offsets;
}

/* create weights */
memory create_weights(engine eng, emb_params &params) {

    int32_t &num_indices = params.num_indices;
    auto weights = memory({{num_indices},
        memory::data_type::f32,
        memory::format_tag::a}, eng);

    float *hndl = (float *)weights.get_data_handle();

    /* weights are in the range [0.5-1.0] */
    for(auto i = 0; i < num_indices; ++i) {
        hndl[i]  = params.weights[i];
    }

    return weights;

}

/* create output */
memory create_output(engine eng, emb_params &params) {

    int32_t &num_bags      = params.num_bags;
    int32_t &dim_embedding = params.dim_embedding;

    /* create destination buffer */
    auto dst = memory({{num_bags, dim_embedding},
        memory::data_type::f32,
        memory::format_tag::ab}, eng);

    return dst;
}

/* sum the bags for verification */
std::vector<float> sum_bags(memory &dst)
{
    std::vector<float> sum;

    auto dst_dims       = dst.get_desc().dims();
    auto num_bags       = dst_dims[0];
    auto dim_embedding  = dst_dims[1];

    float *hndl         = (float *)dst.get_data_handle();

    int32_t idx_base    = 0;
    for(auto bag = 0; bag < num_bags; ++bag) {
        float bag_sum = 0.0;
        for(auto j = 0; j < dim_embedding; ++j) {
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
                        memory bags, algorithm alg,
                        bool is_weights, int32_t padding_idx) {

    auto table_md   = memory::desc(table.get_desc());
    auto indices_md = memory::desc(indices.get_desc());
    auto offsets_md = memory::desc(offsets.get_desc());
    auto weights_md = memory::desc(weights.get_desc());
    auto bags_md    = memory::desc(bags.get_desc());

    auto emdb_d = embedding_bag::desc();

    if(!is_weights) {
        if(padding_idx < 0)
            emdb_d = embedding_bag::desc(prop_kind::forward_inference, alg,
                                         table_md,
                                         indices_md,
                                         offsets_md,
                                         bags_md);
        else
            emdb_d = embedding_bag::desc(prop_kind::forward_inference, alg,
                                         table_md,
                                         indices_md,
                                         offsets_md,
                                         bags_md,
                                         padding_idx);
    } else {
        if(padding_idx < 0)
            emdb_d = embedding_bag::desc(prop_kind::forward_inference, alg,
                                         table_md,
                                         indices_md,
                                         offsets_md,
                                         weights_md,
                                         bags_md);
        else
            emdb_d = embedding_bag::desc(prop_kind::forward_inference, alg,
                                         table_md,
                                         indices_md,
                                         offsets_md,
                                         weights_md,
                                         bags_md,
                                         padding_idx);
    }

    auto emdb_pd = embedding_bag::primitive_desc(emdb_d, eng);
    auto emdb    = embedding_bag(emdb_pd);

    if(!is_weights) {
        emdb.execute(s, {{ZENDNN_ARG_SRC_0, table},
            {ZENDNN_ARG_SRC_1, indices},
            {ZENDNN_ARG_SRC_2, offsets},
            {ZENDNN_ARG_DST, bags}
        });
    } else {
        emdb.execute(s, {{ZENDNN_ARG_SRC_0, table},
            {ZENDNN_ARG_SRC_1, indices},
            {ZENDNN_ARG_SRC_2, offsets},
            {ZENDNN_ARG_SRC_3, weights},
            {ZENDNN_ARG_DST, bags}
        });
    }
}

int main(int argc, char **argv) {

    int status = API_SUCCESS;

    zendnnInfo(ZENDNN_TESTLOG, "ZenDNN API test for embedding_bag starts");

    /* create engine kind */
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    engine eng(engine_kind, 0);
    zendnnVerbose(ZENDNN_TESTLOG, "cpu engine created");

    /* create stream */
    stream s(eng);
    zendnnVerbose(ZENDNN_TESTLOG, "stream created");

    /* get the parameters */
    emb_params params;

    /* create embedding table */
    memory table = create_embedding_table(eng, params);

    /* create indices */
    memory indices = create_indices(eng, params);

    /* create offsets */
    memory offsets = create_offsets(eng, params);

    /* create weights */
    memory weights = create_weights(eng, params);

    /* create output */
    memory bags    = create_output(eng, params);

    zendnnVerbose(ZENDNN_TESTLOG, "all memory buffers created");

    compare_float_t cmp;

    /* test embedding bag */
    {
        zendnnVerbose(ZENDNN_TESTLOG,
                      "testing sum with weights and pading index");
        exec_embedding_bag(eng, s, table, indices,
                           offsets, weights, bags,
                           algorithm::embedding_bag_sum, true,
                           params.padding_idx);

        auto sum = sum_bags(bags);
        for(int i = 0; i < params.num_bags; ++i) {
            if(!cmp(sum[i],expected_output_sum_wt_pd[i])) {
                zendnnError(ZENDNN_TESTLOG, "Expected:",
                            expected_output_sum_wt_pd[i],
                            " Actual:", sum[i]);
                status = API_FAILURE;
            }
        }
    }

    {
        zendnnVerbose(ZENDNN_TESTLOG,
                      "testing mean with no weights, no pading index");
        exec_embedding_bag(eng, s, table, indices,
                           offsets, weights, bags,
                           algorithm::embedding_bag_mean, false, -1);

        auto sum = sum_bags(bags);
        for(int i = 0; i < params.num_bags; ++i) {
            if(!cmp(sum[i],expected_output_mean_nwt_npd[i])) {
                zendnnError(ZENDNN_TESTLOG, "Expected:",
                            expected_output_mean_nwt_npd[i],
                            " Actual:", sum[i]);
                status = API_FAILURE;
            }
        }
    }

    {
        zendnnVerbose(ZENDNN_TESTLOG,
                      "testing max with no weights but padding index");
        exec_embedding_bag(eng, s, table, indices,
                           offsets, weights, bags,
                           algorithm::embedding_bag_max, false,
                           params.padding_idx);

        auto sum = sum_bags(bags);
        for(int i = 0; i < params.num_bags; ++i) {
            if(!cmp(sum[i],expected_output_max_nwt_pd[i])) {
                zendnnError(ZENDNN_TESTLOG, "Expected:",
                            expected_output_max_nwt_pd[i],
                            " Actual:", sum[i]);
                status = API_FAILURE;
            }
        }
    }

    zendnnInfo(ZENDNN_TESTLOG, "ZenDNN API test for embedding_bag ends");
    return status;
}
