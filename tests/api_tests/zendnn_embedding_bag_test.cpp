/*******************************************************************************
* Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

using namespace std;
using namespace zendnn;

/* add new primitive */
/* emable writing output to files */
#define    WRITE_OUTPUT_FILE  1

/* use same seed each time for same random number sequence */
#define    RAND_SEED          (1760)

/* parameters */
#define    NUM_EMBEDDING      (2048)
#define    DIM_EMBEDDING      (10)
#define    NUM_INDICES        (256)
#define    NUM_BAGS           (8)
#define    PADDING_INDEX      (23)

/* create embedding table */
memory create_embedding_table(engine eng, int num_embedding, int dim_embedding) {
    auto table = memory({{num_embedding, dim_embedding},
        memory::data_type::f32,
        memory::format_tag::ab}, eng);

    float *hndl = (float *)table.get_data_handle();

    for (auto i = 0; i < num_embedding*dim_embedding; ++i) {
        hndl[i] = float(rand())/float(num_embedding);
    }

    return table;
}

/* create indices */
memory create_indices(engine eng, int num_indices, int num_embedding) {
    auto indices = memory({{num_indices},
        memory::data_type::s32,
        memory::format_tag::a}, eng);

    int *hndl = (int *)indices.get_data_handle();

    /* indices should be in the range of 0 to num_embedding */
    for (auto i = 0; i < num_indices; ++i) {
        hndl[i] = rand() % num_embedding;
    }

    return indices;
}

/* create offsets */
memory create_offsets(engine eng, int num_bags, int num_indices) {
    auto offsets = memory({{num_bags},
        memory::data_type::s32,
        memory::format_tag::a}, eng);

    int *hndl = (int *)offsets.get_data_handle();

    /* divide indices into equal intevals */
    int intrval = num_indices/(num_bags + 1);

    if (intrval < 1) {
        zendnnError(ZENDNN_TESTLOG, "error in create_offsets,",
                    "offset inverval should at least be 1");
        exit(1);
    }

    /* first offset is always zero */
    hndl[0] = 0;
    for (auto i = 1; i < num_bags; ++i) {
        hndl[i] = hndl[i-1] + intrval;
    }

    return offsets;
}

/* create weights */
memory create_weights(engine eng, int num_indices) {
    auto weights = memory({{num_indices},
        memory::data_type::f32,
        memory::format_tag::a}, eng);

    int *hndl = (int *)weights.get_data_handle();

    /* weights are in the range [0.5-1.0] */
    for (auto i = 0; i < num_indices; ++i) {
        auto r   = float(rand())/float(RAND_MAX);
        hndl[i]  = (r > 0.5) ? r : r + 0.5;
    }

    return weights;

}

/* create output */
memory create_output(engine eng, int num_bags, int dim_embedding) {
    /* create destination buffer */
    auto dst = memory({{num_bags, dim_embedding},
        memory::data_type::f32,
        memory::format_tag::ab}, eng);

    return dst;
}

/* write output to a file */
void write_to_file(string fname, memory mem) {
    ofstream file;

    file.open(fname);
    if (!file) {
        zendnnError(ZENDNN_TESTLOG, "failed to open ", fname);
        return;
    }

    /* get the memory data type */
    size_t size = mem.get_desc().get_size();

    memory::data_type dtype = mem.get_desc().data_type();

    switch (dtype) {
    case memory::data_type::f32: {
        size /= sizeof(float);
        float *hndl = (float *)mem.get_data_handle();
        for (auto i = 0; i < size; ++i) {
            file << hndl[i] << "\n";
        }
    }
    break;

    case memory::data_type::s32: {
        size /= sizeof(int);
        int *hndl = (int *)mem.get_data_handle();
        for (auto i = 0; i < size; ++i) {
            file << hndl[i] << " ";
        }
    }
    break;

    default:
        zendnnInfo(ZENDNN_TESTLOG,
                   "unsupported data type for file write to ", fname);
    }

    file.close();
}

/* execute embedding_bag */
void exec_embedding_bag(engine eng, stream s, memory table,
                        memory indices, memory offsets, memory weights,
                        memory bags, algorithm alg,
                        bool is_weights, bool is_padding_idx) {

    auto table_md   = memory::desc(table.get_desc());
    auto indices_md = memory::desc(indices.get_desc());
    auto offsets_md = memory::desc(offsets.get_desc());
    auto weights_md = memory::desc(weights.get_desc());
    auto bags_md    = memory::desc(bags.get_desc());

    auto emdb_d = embedding_bag::desc();

    if (!is_weights) {
        if (!is_padding_idx)
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
                                         PADDING_INDEX);
    } else {
        if (!is_padding_idx)
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
                                         PADDING_INDEX);
    }

    auto emdb_pd = embedding_bag::primitive_desc(emdb_d, eng);
    auto emdb    = embedding_bag(emdb_pd);

    if (!is_weights) {
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

    /* if enabled write output to file */
#if WRITE_OUTPUT_FILE
    string fname = string(getenv("ZENDNN_GIT_ROOT")) +
                   string("/_out/tests/ref_embeding_bag_out");
    switch (alg) {
    case algorithm::embedding_bag_sum:
        fname += "_sum";
        break;
    case algorithm::embedding_bag_mean:
        fname += "_mean";
        break;
    case algorithm::embedding_bag_max:
        fname += "_max";
        break;
    default:
        zendnnInfo(ZENDNN_TESTLOG, "unknown algorithm type");
    }

    if (is_weights) {
        fname += "_w";
    }
    if (is_padding_idx) {
        fname += "_pd";
    }

    zendnnInfo(ZENDNN_TESTLOG, "writing output to ", fname);

    write_to_file(fname, bags);
#endif

}

int main(int argc, char **argv) {
    zendnnInfo(ZENDNN_TESTLOG, "ZenDNN API test for embedding_bag starts");

    /* create engine kind */
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    engine eng(engine_kind, 0);
    zendnnVerbose(ZENDNN_TESTLOG, "cpu engine created");

    /* create stream */
    stream s(eng);
    zendnnVerbose(ZENDNN_TESTLOG, "stream created");

    /* initialize random number generator */
    srand(RAND_SEED);

    /* create embedding table */
    memory table = create_embedding_table(eng, NUM_EMBEDDING, DIM_EMBEDDING);

    /* create indices */
    memory indices = create_indices(eng, NUM_INDICES, NUM_EMBEDDING);

    /* create offsets */
    memory offsets = create_offsets(eng, NUM_BAGS, NUM_INDICES);

    /* create weights */
    memory weights = create_weights(eng, NUM_INDICES);

    /* create output */
    memory bags    = create_output(eng, NUM_BAGS, DIM_EMBEDDING);

    zendnnVerbose(ZENDNN_TESTLOG, "all memory buffers created");

    /* test embedding bag without weights and padding index */
    exec_embedding_bag(eng, s, table, indices,
                       offsets, weights, bags,
                       algorithm::embedding_bag_sum, false, true);
    exec_embedding_bag(eng, s, table, indices,
                       offsets, weights, bags,
                       algorithm::embedding_bag_mean, false, true);
    exec_embedding_bag(eng, s, table, indices,
                       offsets, weights, bags,
                       algorithm::embedding_bag_max, false, true);

    zendnnInfo(ZENDNN_TESTLOG, "ZenDNN API test for embedding_bag ends");
    return 0;

}
