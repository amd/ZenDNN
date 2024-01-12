/*******************************************************************************
* Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "zendnn.hpp"
#include "zendnn_helper.hpp"
#include <vector>
#include <omp.h>
#define ZENDNN_EMBED_BAG_THRDS 16
#define CCD_NUM_THREADS 8
namespace zendnn {

void zendnn_embedding_bag_exec(
    const memory &z_input, const memory &z_indices, const memory &z_offsets,
    const int32_t &scale_grad_by_freq,
    const algorithm &z_algorithm, const int32_t &sparse,
    const memory &z_per_sample_weights,
    const int32_t &z_per_sample_weights_defined,
    const int32_t &include_last_offset, const int32_t &padding_idx, memory &z_dst,
    unsigned int op_num_threads=1) {

    engine eng;
    stream s;
    eng=engine(engine::kind::cpu, 0);
    s=stream(eng);

    embedding_bag::desc pdesc;
    embedding_bag::primitive_desc pd;

    if (z_per_sample_weights_defined) {

        // declare embedding bag primitive
        pdesc = embedding_bag::desc(prop_kind::forward_inference,
                                    z_algorithm,
                                    op_num_threads,
                                    z_input.get_desc(),
                                    z_indices.get_desc(),
                                    z_offsets.get_desc(),
                                    z_per_sample_weights.get_desc(),
                                    z_dst.get_desc(),
                                    padding_idx);

        pd = embedding_bag::primitive_desc(pdesc, eng);

        embedding_bag(pd).execute(s, {{ZENDNN_ARG_SRC_0, z_input},
            {ZENDNN_ARG_SRC_1, z_indices},
            {ZENDNN_ARG_SRC_2, z_offsets},
            {ZENDNN_ARG_SRC_3, z_per_sample_weights},
            {ZENDNN_ARG_DST, z_dst}
        });
    }
    else {
        // declare embedding bag primitive
        pdesc = embedding_bag::desc(prop_kind::forward_inference,
                                    z_algorithm,
                                    ZENDNN_EMBED_BAG_THRDS,
                                    z_input.get_desc(),
                                    z_indices.get_desc(),
                                    z_offsets.get_desc(),
                                    z_dst.get_desc(),
                                    padding_idx);

        pd = embedding_bag::primitive_desc(pdesc, eng);

        embedding_bag(pd).execute(s, {{ZENDNN_ARG_SRC_0, z_input},
            {ZENDNN_ARG_SRC_1, z_indices},
            {ZENDNN_ARG_SRC_2, z_offsets},
            {ZENDNN_ARG_DST, z_dst}
        });
    }
}

void zendnn_embedding_exec(
    const memory &z_input, const memory &z_indices,
    const int32_t &padding_idx, const int32_t &scale_grad_by_freq,
    const int32_t &z_sparse, memory &z_dst, int op_num_threads=1) {

    algorithm z_algorithm = algorithm::embedding_bag_sum;

    engine eng;
    stream s;
    eng=engine(engine::kind::cpu, 0);
    s=stream(eng);

    const auto &mem = z_indices;
    int size = static_cast<int>(memory::desc(mem.get_desc()).get_size()/sizeof(
                                    int32_t));
    // Create offset memory
    auto z_offsets = memory({{size},
        memory::data_type::s32,
        memory::format_tag::a}, eng);

    // Assign values in offsets
    int32_t *hndl = static_cast<int32_t *>(z_offsets.get_data_handle());
    zendnnEnv zenEnvObj = readEnv();
    unsigned int nthrds = zenEnvObj.omp_num_threads;

    #pragma omp parallel for num_threads(nthrds)
    for (int j = 0; j < size; ++j) {
        hndl[j] = j;
    }

//TODO New constructor need to be defined for embedding op

    embedding_bag::desc pdesc;
    embedding_bag::primitive_desc pd;

    // declare embedding bag primitive
    pdesc = embedding_bag::desc(prop_kind::forward_inference,
                                z_algorithm,
                                ZENDNN_EMBED_BAG_THRDS,
                                z_input.get_desc(),
                                z_indices.get_desc(),
                                z_offsets.get_desc(),
                                z_dst.get_desc(),
                                padding_idx);

    pd = embedding_bag::primitive_desc(pdesc, eng);

    embedding_bag(pd).execute(s, {{ZENDNN_ARG_SRC_0, z_input},
        {ZENDNN_ARG_SRC_1, z_indices},
        {ZENDNN_ARG_SRC_2, z_offsets},
        {ZENDNN_ARG_DST, z_dst}
    });
}

//API call to perform embedding lookups on bags of indices and then optionally apply
// a reduction opration(such as sum, mean or max) on the embedding within each bag.

void zendnn_custom_op::zendnn_embedding_bag(const memory &z_input,
        const memory &z_indices,
        const memory &z_offsets,
        const bool &z_scale_grad_by_freq,
        const algorithm &z_mode, const bool &z_sparse,
        const memory &z_per_sample_weights_opt,
        const bool &z_per_sample_weights_defined,
        const bool &z_include_last_offset, const int32_t &z_padding_idx,
        memory &z_destination, int thread_qty) {

    zendnn_embedding_bag_exec(
        z_input, z_indices, z_offsets, static_cast<int32_t>(z_scale_grad_by_freq),
        z_mode,
        static_cast<int32_t>(z_sparse), z_per_sample_weights_opt,
        static_cast<int32_t>(z_per_sample_weights_defined),
        static_cast<int32_t>(z_include_last_offset),
        z_padding_idx, z_destination);
}

void zendnn_custom_op::zendnn_grp_embedding_bag(std::vector <memory> &z_input,
        std::vector <memory> &z_indices, std::vector <memory> &z_offsets,
        std::vector <int32_t> &z_scale_grad_by_freq, std::vector <algorithm> &z_modes,
        std::vector <int32_t> &z_sparse, std::vector <memory> &z_per_sample_weights_opt,
        std::vector <int32_t> &z_per_sample_weights_defined,
        std::vector <int32_t> &z_include_last_offset,
        std::vector <int32_t> &z_padding_idx,
        std::vector <memory> &z_destination, int thread_qty) {

    zendnnEnv zenEnvObj = readEnv();
    unsigned int eb_thread_qty = zenEnvObj.omp_num_threads;
    int num_tables = z_input.size();

    if (zenEnvObj.zenEBAlgo==zenEBAlgoType::CCD_THREADED) {
        omp_set_max_active_levels(2);
        int ccd_num_threads=CCD_NUM_THREADS;
        unsigned int outer_threads = (eb_thread_qty%ccd_num_threads)==0 ?
                                     eb_thread_qty/ccd_num_threads: ((eb_thread_qty/ccd_num_threads)+1);
        unsigned int rem = (eb_thread_qty%ccd_num_threads)==0 ? ccd_num_threads :
                           eb_thread_qty%ccd_num_threads;
        unsigned int loopCount = (num_tables%outer_threads)==0 ?
                                 num_tables/outer_threads : ((num_tables/outer_threads)+1);

        #pragma omp parallel num_threads(outer_threads)
        {
            unsigned int inner_threads = ccd_num_threads;
            unsigned int thid = omp_get_thread_num();
            if (thid == outer_threads-1) {
                inner_threads = rem;
            }

            for (int i=0; i<loopCount; i++) {
                int threadOffset = thid+ (i*outer_threads);
                if (threadOffset >= num_tables) {
                    break;
                }

                zendnn_embedding_bag_exec(
                    z_input[threadOffset], z_indices[threadOffset], z_offsets[threadOffset],
                    z_scale_grad_by_freq[threadOffset], z_modes[threadOffset],
                    z_sparse[threadOffset], z_per_sample_weights_opt[threadOffset],
                    z_per_sample_weights_defined[threadOffset],
                    z_include_last_offset[threadOffset],
                    z_padding_idx[threadOffset], z_destination[threadOffset],inner_threads);
            }

        }
    }
    else if (num_tables<eb_thread_qty &&
             zenEnvObj.zenEBAlgo==zenEBAlgoType::HYBRID_THREADED) {
        unsigned int outer_threads = num_tables;
        unsigned int rem = eb_thread_qty%num_tables;
        #pragma omp parallel num_threads(outer_threads)
        {
            unsigned int inner_threads = eb_thread_qty/num_tables;
            unsigned int threadOffset = omp_get_thread_num();
            if (threadOffset < rem) {
                inner_threads++;
            }
            zendnn_embedding_bag_exec(
                z_input[threadOffset], z_indices[threadOffset], z_offsets[threadOffset],
                z_scale_grad_by_freq[threadOffset], z_modes[threadOffset],
                z_sparse[threadOffset], z_per_sample_weights_opt[threadOffset],
                z_per_sample_weights_defined[threadOffset],
                z_include_last_offset[threadOffset],
                z_padding_idx[threadOffset], z_destination[threadOffset],inner_threads);
        }
    }

    else if (zenEnvObj.zenEBAlgo==zenEBAlgoType::BATCH_THREADED) {
        for (int i = 0; i < num_tables; i++) {
            zendnn_embedding_bag_exec(
                z_input[i], z_indices[i], z_offsets[i],
                z_scale_grad_by_freq[i], z_modes[i],
                z_sparse[i], z_per_sample_weights_opt[i],z_per_sample_weights_defined[i],
                z_include_last_offset[i],
                z_padding_idx[i], z_destination[i]);
        }

    }

    else {
        unsigned int loopCount = (num_tables%eb_thread_qty)==0 ?
                                 num_tables/eb_thread_qty : ((num_tables/eb_thread_qty)+1);
        #pragma omp parallel num_threads(eb_thread_qty)
        {
            for (int i=0; i<loopCount; i++) {
                int threadOffset = omp_get_thread_num()+ (i*eb_thread_qty);
                if (threadOffset >= num_tables) {
                    break;
                }
                zendnn_embedding_bag_exec(
                    z_input[threadOffset], z_indices[threadOffset], z_offsets[threadOffset],
                    z_scale_grad_by_freq[threadOffset], z_modes[threadOffset],
                    z_sparse[threadOffset], z_per_sample_weights_opt[threadOffset],
                    z_per_sample_weights_defined[threadOffset],
                    z_include_last_offset[threadOffset],
                    z_padding_idx[threadOffset], z_destination[threadOffset]);
            }
        }

    }
}

//API call to perform just embedding lookup where each input index corresponds to single embedding.

void zendnn_custom_op::zendnn_embedding(const memory &z_input,
                                        const memory &z_indices,
                                        const int32_t &z_padding_idx, const bool &z_scale_grad_by_freq,
                                        const bool &z_sparse,
                                        memory &z_destination, int thread_qty) {

    zendnn_embedding_exec(
        z_input, z_indices, z_padding_idx, static_cast<int32_t>(z_scale_grad_by_freq),
        static_cast<int32_t>(z_sparse),
        z_destination);
}

void zendnn_custom_op::zendnn_grp_embedding(std::vector <memory> &z_input,
        std::vector <memory> &z_indices,
        std::vector <int32_t> &z_padding_idx,
        std::vector <int32_t> &z_scale_grad_by_freq,
        std::vector <int32_t> &z_sparse,
        std::vector <memory> &z_destination, int thread_qty) {


    for (int i = 0; i < z_input.size(); i++) {
        zendnn_embedding_exec(
            z_input[i], z_indices[i], z_padding_idx[i], z_scale_grad_by_freq[i],
            z_sparse[i], z_destination[i]);
    }
}
}//ZenDNN

