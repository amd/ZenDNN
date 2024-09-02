/*******************************************************************************
* Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <string.h>
#include "zendnn_logging.hpp"
#include "common/verbose.hpp"
#define ZENDNN_EMBED_BAG_THRDS 16
#define CCD_NUM_THREADS 8
#if FBGEMM_ENABLE
    #include "fbgemm/FbgemmEmbedding.h"
    using namespace fbgemm;
#endif
namespace zendnn {

void zendnn_embedding_bag_kernel(
    const memory &z_input, const memory &z_indices, const memory &z_offsets,
    const int32_t &scale_grad_by_freq,
    const algorithm &z_algorithm, const int32_t &sparse,
    const memory &z_per_sample_weights,
    const int32_t &z_per_sample_weights_defined,
    const int32_t &include_last_offset, const int32_t &padding_idx, memory &z_dst,
    unsigned int op_num_threads, const char *plugin_op) {
    engine eng;
    stream s;
    eng=engine(engine::kind::cpu, 0);
    s=stream(eng);
    primitive_attr op_attr;
    std::string op_name=plugin_op;
    op_attr.set_plugin_op_name(op_name);

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

        pd = embedding_bag::primitive_desc(pdesc, op_attr, eng);

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
                                    op_num_threads,
                                    z_input.get_desc(),
                                    z_indices.get_desc(),
                                    z_offsets.get_desc(),
                                    z_dst.get_desc(),
                                    padding_idx);

        pd = embedding_bag::primitive_desc(pdesc, op_attr, eng);

        embedding_bag(pd).execute(s, {{ZENDNN_ARG_SRC_0, z_input},
            {ZENDNN_ARG_SRC_1, z_indices},
            {ZENDNN_ARG_SRC_2, z_offsets},
            {ZENDNN_ARG_DST, z_dst}
        });
    }
}

void zendnn_embedding_bag_exec(
    const memory &z_input, const memory &z_indices, const memory &z_offsets,
    const int32_t &scale_grad_by_freq,
    const algorithm &z_algorithm, const int32_t &sparse,
    const memory &z_per_sample_weights,
    const int32_t &z_per_sample_weights_defined,
    const int32_t &include_last_offset, const int32_t &padding_idx, memory &z_dst,
    const char *plugin_op, unsigned int op_num_threads) {

    zendnnEnv EnvObj = readEnv();
    auto emd_table_dims = z_input.get_desc().dims();
    auto dim_embedding  = emd_table_dims[1];
    auto num_rows       = emd_table_dims[0];
    int indices_size    = z_indices.get_desc().dims()[0];
    int batch_size      = z_dst.get_desc().dims()[0];

#if FBGEMM_ENABLE
    // FBGEMM kernel path is taken for algo sum
    if (EnvObj.zenEBAlgo==zenEBAlgoType::EB_OP_FBGEMM &&
            z_algorithm==algorithm::embedding_bag_sum) {
        double start_ms = impl::get_msec();

        float *table_ptr = static_cast<float *>(z_input.get_data_handle());
        int32_t *indices = static_cast<int32_t *>(z_indices.get_data_handle());
        int32_t *offsets = static_cast<int32_t *>(z_offsets.get_data_handle());
        float *output    = static_cast<float *>(z_dst.get_data_handle());

        bool use_weight=false;
        bool normalize_by_lengths=false;
        bool prefetch=true;
        bool is_wt_positional=false;
        bool use_offsets=true;
        bool ret;
        auto kernel = GenerateEmbeddingSpMDM<float, int32_t>(
                          dim_embedding,
                          use_weight,
                          normalize_by_lengths,
                          prefetch?16:0,
                          is_wt_positional,
                          use_offsets);

        if (include_last_offset==0) {
            int32_t *fbgemm_offsets = new int32_t[batch_size+1];
            memcpy(fbgemm_offsets, offsets, batch_size * sizeof(int32_t));
            fbgemm_offsets[batch_size]=indices_size;

            ret = kernel(
                      batch_size,
                      indices_size,
                      num_rows,
                      table_ptr,
                      indices,
                      (const int *)fbgemm_offsets,
                      nullptr,
                      output);

            delete[] fbgemm_offsets;
        }
        else {
            ret = kernel(
                      batch_size,
                      indices_size,
                      num_rows,
                      table_ptr,
                      indices,
                      (const int *)offsets,
                      nullptr,
                      output);
        }

        double duration_ms = impl::get_msec() - start_ms;

        zendnnVerbose(ZENDNN_PROFLOG, "zendnn_primitive_execute,cpu,embedding_bag,",
                      "fbgemm",",","BS:",batch_size,",ED:",dim_embedding,",alg:sum",
                      ",",duration_ms,
                      ",ms");
    }
    else {
        zendnn_embedding_bag_kernel(
            z_input, z_indices, z_offsets,
            scale_grad_by_freq, z_algorithm,
            sparse, z_per_sample_weights,
            z_per_sample_weights_defined,
            include_last_offset,
            padding_idx, z_dst, op_num_threads, plugin_op);
    }
#else
    zendnn_embedding_bag_kernel(
        z_input, z_indices, z_offsets,
        scale_grad_by_freq, z_algorithm,
        sparse, z_per_sample_weights,
        z_per_sample_weights_defined,
        include_last_offset,
        padding_idx, z_dst, op_num_threads, plugin_op);
#endif

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
        memory &z_destination, const char *plugin_op, int thread_qty) {

    zendnnEnv zenEnvObj = readEnv();
    unsigned int eb_thread_qty = zenEnvObj.omp_num_threads;

    zendnn_embedding_bag_exec(
        z_input, z_indices, z_offsets, static_cast<int32_t>(z_scale_grad_by_freq),
        z_mode,
        static_cast<int32_t>(z_sparse), z_per_sample_weights_opt,
        static_cast<int32_t>(z_per_sample_weights_defined),
        static_cast<int32_t>(z_include_last_offset),
        z_padding_idx, z_destination, plugin_op, eb_thread_qty);
}

void zendnn_custom_op::zendnn_grp_embedding_bag(std::vector <memory> &z_input,
        std::vector <memory> &z_indices, std::vector <memory> &z_offsets,
        std::vector <int32_t> &z_scale_grad_by_freq, std::vector <algorithm> &z_modes,
        std::vector <int32_t> &z_sparse, std::vector <memory> &z_per_sample_weights_opt,
        std::vector <int32_t> &z_per_sample_weights_defined,
        std::vector <int32_t> &z_include_last_offset,
        std::vector <int32_t> &z_padding_idx,
        std::vector <memory> &z_destination, const char *plugin_op, int thread_qty) {

    zendnnEnv zenEnvObj = readEnv();
    unsigned int eb_thread_qty = zenEnvObj.omp_num_threads;
    int num_tables = z_input.size();
    std::string thread_type;
    int batch_size      = z_destination[0].get_desc().dims()[0];
    int embedding_dim   = z_input[0].get_desc().dims()[1];

    double start_ms = impl::get_msec();
    if (zenEnvObj.zenEBThreadAlgo==zenEBThreadType::CCD_THREADED) {
        thread_type="ccd_threaded";
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
                    z_padding_idx[threadOffset], z_destination[threadOffset], plugin_op,
                    inner_threads);
            }

        }
    }
    else if (num_tables<eb_thread_qty &&
             zenEnvObj.zenEBThreadAlgo==zenEBThreadType::HYBRID_THREADED) {
        thread_type="hybrid_threaded";
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
                z_padding_idx[threadOffset], z_destination[threadOffset], plugin_op,
                inner_threads);
        }
    }

    else if (zenEnvObj.zenEBThreadAlgo==zenEBThreadType::BATCH_THREADED) {
        thread_type="batch_threaded";
        for (int i = 0; i < num_tables; i++) {
            zendnn_embedding_bag_exec(
                z_input[i], z_indices[i], z_offsets[i],
                z_scale_grad_by_freq[i], z_modes[i],
                z_sparse[i], z_per_sample_weights_opt[i],z_per_sample_weights_defined[i],
                z_include_last_offset[i],
                z_padding_idx[i], z_destination[i], plugin_op, eb_thread_qty);
        }

    }

    else {
        thread_type="table_threaded";
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
                    z_padding_idx[threadOffset], z_destination[threadOffset], plugin_op, 1);
            }
        }

    }
    double duration_ms = impl::get_msec() - start_ms;

    zendnnVerbose(ZENDNN_PROFLOG, "zendnn_custom_op_execute,cpu,plugin_op:",
                  plugin_op, ",","num_ops:",num_tables,",","dims:batch_size=",batch_size,",",
                  "alg:",thread_type,",",duration_ms,",ms");
}

//API call to perform just embedding lookup where each input index corresponds to single embedding.

void zendnn_custom_op::zendnn_embedding(const memory &z_input,
                                        const memory &z_indices,
                                        const int32_t &z_padding_idx, const bool &z_scale_grad_by_freq,
                                        const bool &z_sparse,
                                        memory &z_destination, const char *plugin_op, int thread_qty) {

    int indices_size = z_indices.get_desc().dims()[0];
    int32_t z_per_sample_weights_defined=0;
    int32_t z_include_last_offset=0;

    engine eng;
    eng=engine(engine::kind::cpu, 0);
    auto z_offsets=memory({{indices_size},
        memory::data_type::s32,
        memory::format_tag::a}, eng);
    int32_t *hndl = static_cast<int32_t *>(z_offsets.get_data_handle());

    zendnnEnv zenEnvObj = readEnv();
    unsigned int num_thread = zenEnvObj.omp_num_threads;

    #pragma omp parallel for num_threads(num_thread)
    for (int k = 0; k < indices_size; k++) {
        hndl[k] = k;
    }

    algorithm z_mode=algorithm::embedding_bag_sum;
    auto z_per_sample_weights_opt=memory({{indices_size},
        memory::data_type::s32,
        memory::format_tag::a}, eng, nullptr);

    zendnn_embedding_bag_exec(
        z_input, z_indices, z_offsets, static_cast<int32_t>(z_scale_grad_by_freq),
        z_mode,
        static_cast<int32_t>(z_sparse), z_per_sample_weights_opt,
        static_cast<int32_t>(z_per_sample_weights_defined),
        static_cast<int32_t>(z_include_last_offset),
        z_padding_idx, z_destination, plugin_op, num_thread);
}

void zendnn_custom_op::zendnn_grp_embedding(std::vector <memory> &z_input,
        std::vector <memory> &z_indices,
        std::vector <int32_t> &z_padding_idx,
        std::vector <int32_t> &z_scale_grad_by_freq,
        std::vector <int32_t> &z_sparse,
        std::vector <memory> &z_destination, const char *plugin_op, int thread_qty) {

    int num_eb_ops = z_input.size();
    std::vector <memory> z_offsets(num_eb_ops);
    std::vector <algorithm> z_modes(num_eb_ops);
    std::vector <memory> z_per_sample_weights_opt(num_eb_ops);
    std::vector <int32_t> z_per_sample_weights_defined(num_eb_ops,0);
    std::vector <int32_t> z_include_last_offset(num_eb_ops,0);

    zendnnEnv zenEnvObj = readEnv();
    unsigned int num_thread = zenEnvObj.omp_num_threads;
    engine eng;
    eng=engine(engine::kind::cpu, 0);

    #pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i <  num_eb_ops; i++) {

        int indices_size = z_indices[i].get_desc().dims()[0];
        z_offsets[i] = memory({{indices_size},
            memory::data_type::s32,
            memory::format_tag::a}, eng);
        int32_t *hndl = static_cast<int32_t *>(z_offsets[i].get_data_handle());

        for (int k = 0; k < indices_size; k++) {
            hndl[k] = k;
        }

        z_per_sample_weights_opt[i] = memory({{indices_size},
            memory::data_type::s32,
            memory::format_tag::a}, eng, nullptr);

        z_modes[i]=algorithm::embedding_bag_sum;
    }

    zendnn_custom_op::zendnn_grp_embedding_bag(
        z_input, z_indices, z_offsets, z_scale_grad_by_freq, z_modes,
        z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
        z_include_last_offset, z_padding_idx, z_destination, plugin_op, thread_qty);

}
}//ZenDNN

