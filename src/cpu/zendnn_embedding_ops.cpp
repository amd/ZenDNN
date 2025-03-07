/*******************************************************************************
* Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <type_traits>
#include "zendnn_logging.hpp"
#include "common/verbose.hpp"
#include <cstddef>
#include <exception>
#define ZENDNN_EMBED_BAG_THRDS 16
#define CCD_NUM_THREADS 8

/*Both scale and Bias are in Float16,
so number of s4 elements are 8*/
#define SCALE_BIAS_SIZE 8
#define ZENDNN_PARALLEL_FOR 1
#define ZENDNN_PARALLEL_FOR_TT 1
#if FBGEMM_ENABLE
    #include "fbgemm/FbgemmEmbedding.h"
#endif
namespace zendnn {

inline int64_t divup(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

template <class F>
inline void zendnn_parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F &f) {
    if (grain_size >= 0);
    if (begin >= end) {
        return;
    }
    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;
    // choose number of tasks based on grain size and number of threads
    int64_t num_threads = omp_in_parallel() ? 1 : omp_get_max_threads();
    if (grain_size > 0) {
        num_threads = std::min(num_threads, divup((end - begin), grain_size));
    }

    #pragma omp parallel num_threads(num_threads)
    {
        int64_t num_threads = omp_get_num_threads();
        int64_t tid = omp_get_thread_num();
        int64_t chunk_size = divup((end - begin), num_threads);
        int64_t begin_tid = begin + tid * chunk_size;
        if (begin_tid < end) {
            try {
                f(begin_tid, std::min(end, chunk_size + begin_tid));
            }
            catch (...) {
                if (!err_flag.test_and_set()) {
                    eptr = std::current_exception();
                }
            }
        }
    }
    if (eptr) {
        std::rethrow_exception(eptr);
    }
}

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

#if FBGEMM_ENABLE
template<typename IN_TYPE, typename OUT_TYPE>
void fbgemm_embedding_bag_kernel(
    const memory &z_input, const memory &z_indices, const memory &z_offsets,
    const int32_t &scale_grad_by_freq,
    const algorithm &z_algorithm, const int32_t &sparse,
    const memory &z_per_sample_weights,
    const int32_t &z_per_sample_weights_defined,
    const int32_t &include_last_offset, const int32_t &padding_idx, memory &z_dst,
    unsigned int op_num_threads, const char *plugin_op,
    const bool &scale_bias_last, const int &cat_dim, const int &mlp_pos,
    const int &output_stride, const int idx,
    const int num_tables) {

    //double start_ms = impl::get_msec();
    auto emd_table_dims = z_input.get_desc().dims();
    auto dim_embedding  = (z_input.get_desc().data_type()==impl::data_type::s4)?
                          z_input.get_desc().dims()[1]-SCALE_BIAS_SIZE : z_input.get_desc().dims()[1];
    auto num_rows       = emd_table_dims[0];
    int indices_size    = z_indices.get_desc().dims()[0];
    int batch_size      = (include_last_offset==0)?z_offsets.get_desc().dims()[0] :
                          z_offsets.get_desc().dims()[0]-1;
    int bit_rate;
    int stride_offset;
    bool is_bf16_out;
    bool is_bf16_in;
    bool use_weight=z_per_sample_weights_defined;
    bool normalize_by_lengths=false;
    bool prefetch=true;
    bool is_wt_positional=false;
    bool use_offsets=true;
    std::string in_dtype;
    std::string out_dtype;
    IN_TYPE *table_ptr=NULL;
    float *weights=nullptr;
    OUT_TYPE *output=NULL;
    OUT_TYPE *cat_output=NULL;
    int32_t *fbgemm_offsets=nullptr;
    int32_t *indices = static_cast<int32_t *>(z_indices.get_data_handle());
    int32_t *offsets = static_cast<int32_t *>(z_offsets.get_data_handle());

    if (include_last_offset==0) {
        fbgemm_offsets = new int32_t[batch_size+1];
        memcpy(fbgemm_offsets, offsets, batch_size * sizeof(int32_t));
        static_cast<int32_t *>(fbgemm_offsets)[batch_size]=indices_size;
    }
    else {
        fbgemm_offsets=offsets;
    }

    //TODO: ADD INT8 Data type for Embedding table
    if ((z_input.get_desc().data_type()==impl::data_type::bf16) &&
            (z_dst.get_desc().data_type()==impl::data_type::bf16)) {
        is_bf16_in=true;
        is_bf16_out=true;
        in_dtype="src_bf16";
        out_dtype="dst_bf16";
    }
    else if ((z_input.get_desc().data_type()==impl::data_type::bf16) &&
             (z_dst.get_desc().data_type()==impl::data_type::f32)) {
        is_bf16_in=true;
        is_bf16_out=false;
        in_dtype="src_bf16";
        out_dtype="dst_f32";
    }
    else if ((z_input.get_desc().data_type()==impl::data_type::f32) &&
             (z_dst.get_desc().data_type()==impl::data_type::f32)) {
        is_bf16_in=false;
        is_bf16_out=false;
        in_dtype="src_f32";
        out_dtype="dst_f32";
    }
    else if ((z_input.get_desc().data_type()==impl::data_type::s4) &&
             (z_dst.get_desc().data_type()==impl::data_type::bf16)) {
        bit_rate=4;
        is_bf16_out=true;
        in_dtype="src_s4";
        out_dtype="dst_bf16";
    }
    else if ((z_input.get_desc().data_type()==impl::data_type::s4) &&
             (z_dst.get_desc().data_type()==impl::data_type::f32)) {
        bit_rate=4;
        is_bf16_out=false;
        in_dtype="src_s4";
        out_dtype="dst_f32";
    }
    else {
        ZENDNN_THROW_ERROR(zendnn_invalid_arguments, "Data Type not Supported");
    }

    table_ptr = static_cast<IN_TYPE *>(z_input.get_data_handle());
    output    = static_cast<OUT_TYPE *>(z_dst.get_data_handle());
    if (use_weight==true) {
        weights   = static_cast<float *>(z_per_sample_weights.get_data_handle());
    }

    if (cat_dim==1) {
        stride_offset = output_stride;
        int mlp_offset = output_stride - num_tables*dim_embedding;
        cat_output = (mlp_pos==0) ? output + (idx*dim_embedding) + mlp_offset:
                     output + (idx*dim_embedding);
    }
    else if (cat_dim==0) {
        stride_offset = dim_embedding;
        int mlp_offset = (z_dst.get_desc().dims()[0]-batch_size*num_tables)
                         *dim_embedding;
        cat_output = (mlp_pos==0) ? output + mlp_offset + (idx*batch_size*dim_embedding)
                     :
                     output + (idx * batch_size * dim_embedding);
    }
    else {
        stride_offset = dim_embedding;
        cat_output=output;
    }

    using KernelType = typename std::conditional<
                       std::is_same<IN_TYPE, uint8_t>::value,
                       typename fbgemm::EmbeddingSpMDMKernelSignature<uint8_t, int, std::int32_t, OUT_TYPE>::Type,
                       typename fbgemm::EmbeddingSpMDMKernelSignature<IN_TYPE, int, std::int32_t, OUT_TYPE>::Type>::type;

    KernelType kernel;

    if constexpr(std::is_same<IN_TYPE, uint8_t>::value) {
        kernel =
            fbgemm::GenerateEmbeddingSpMDMNBitWithStrides<int32_t, int32_t,
            OUT_TYPE,
            true/*TRHEAD_LOCAL=*/>(
                bit_rate,
                dim_embedding,
                use_weight,
                normalize_by_lengths,
                prefetch?16:0,
                is_wt_positional,
                use_offsets,
                output_stride,
                -1, /*input_stride=*/
                scale_bias_last,
                is_bf16_out);
    }
    else {
        kernel =
            fbgemm::GenerateEmbeddingSpMDMWithStrides<IN_TYPE, int32_t, int32_t, OUT_TYPE>(
                dim_embedding,
                use_weight,
                normalize_by_lengths,
                prefetch?16:0,
                is_wt_positional,
                use_offsets,
                output_stride,
                -1, /*input_stride=*/
                true, /* scale_bias_last*/
                false, /*no bags*/
                is_bf16_out,
                is_bf16_in);
    }


#if ZENDNN_PARALLEL_FOR
    if (op_num_threads == 1) {
        kernel(
            batch_size,
            indices_size,
            num_rows,
            table_ptr,
            indices,
            (const int *)fbgemm_offsets,
            weights,
            cat_output);
    }
    else {
        zendnn_parallel_for(0, batch_size, 1, [&](int start_idx, int end_idx) {
            bool success = kernel(
                               /*output_size=*/end_idx - start_idx,
                               /*index_size=*/indices_size,
                               /*data_size=*/num_rows,
                               /*input=*/table_ptr,
                               /*indices=*/&indices[fbgemm_offsets[start_idx]],
                               /*offsets_or_lengths=*/(const int *)&fbgemm_offsets [start_idx],
                               /*weights=*/nullptr,
                               /*out=*/&cat_output[start_idx * stride_offset]);

        });
    }
#else
    // TODO: Generate more heuristics based on batch size, pooling size and number of threads.
    // Current decision logic is based on the batch size and number of threads.
    if (op_num_threads == 1 || batch_size <= 100 ||
            (z_input.get_desc().data_type()==impl::data_type::s4 && batch_size <=1024) ||
            (z_input.get_desc().data_type()==impl::data_type::s4 && op_num_threads <= 2)) {
        kernel(
            batch_size,
            indices_size,
            num_rows,
            table_ptr,
            indices,
            (const int *)fbgemm_offsets,
            weights,
            cat_output);
    }
    else {
        #pragma omp parallel for num_threads(op_num_threads)
        for (int i = 0; i < batch_size; ++i) {
            int start_idx = fbgemm_offsets[i];
            int end_idx = fbgemm_offsets[i + 1];
            int current_batch_size = end_idx - start_idx;
            // Call the kernel for the current batch
            kernel(
                current_batch_size,
                indices_size,
                num_rows,
                table_ptr,
                &indices[start_idx],
                (const int *)&fbgemm_offsets[i],
                weights,
                &cat_output[i * stride_offset]);
        }
    }
#endif
    if (include_last_offset==0) {
        delete[] fbgemm_offsets;
    }
    /*double duration_ms = impl::get_msec() - start_ms;
    zendnnVerbose(ZENDNN_PROFLOG, "zendnn_primitive_execute,cpu",",","plugin_op:",
                  plugin_op,",","fbgemm",",",in_dtype,",",out_dtype,",",
                  "BS:",batch_size,",ED:",dim_embedding,",alg:sum",",",duration_ms,
                  ",ms");*/
}
#endif

template<typename IN_TYPE, typename OUT_TYPE>
void zendnn_embedding_bag_exec(
    const memory &z_input, const memory &z_indices, const memory &z_offsets,
    const int32_t &scale_grad_by_freq,
    const algorithm &z_algorithm, const int32_t &sparse,
    const memory &z_per_sample_weights,
    const int32_t &z_per_sample_weights_defined,
    const int32_t &include_last_offset, const int32_t &padding_idx, memory &z_dst,
    const char *plugin_op, unsigned int op_num_threads,
    const bool &scale_bias_last, const int &cat_dim=-1, const int &mlp_pos=-1,
    const int &output_stride=-1, const int idx=0,
    const int num_tables=1) {

    zendnnEnv EnvObj = readEnv();
    int batch_size      = z_dst.get_desc().dims()[0];

#if FBGEMM_ENABLE
// TODO: Generate more heuristics based on batch size and pooling size.
// Current decision logic is based on the batch size and optimal kernel
// available from zendnn and fbgemm

//TODO: Add INT4 support for Mean and Max
    if ((z_input.get_desc().data_type()==impl::data_type::s4) &&
            (z_algorithm!=algorithm::embedding_bag_sum)) {
        ZENDNN_THROW_ERROR(zendnn_invalid_arguments,
                           "Mean & Max algo not Supported for INT4 Data Type");
    }
    if ((EnvObj.zenEBAlgo==zenEBAlgoType::EB_OP_FBGEMM || op_num_threads==1 ||
            batch_size<=100 || cat_dim!=-1 ||
            z_input.get_desc().data_type()==impl::data_type::s4) &&
            (z_algorithm==algorithm::embedding_bag_sum)) {

        fbgemm_embedding_bag_kernel<IN_TYPE, OUT_TYPE>(
            z_input, z_indices, z_offsets, scale_grad_by_freq,
            z_algorithm, sparse, z_per_sample_weights,
            z_per_sample_weights_defined,
            include_last_offset,
            padding_idx, z_dst, op_num_threads, plugin_op, scale_bias_last, cat_dim,
            mlp_pos, output_stride, idx,
            num_tables);
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

template<typename IN_TYPE, typename OUT_TYPE>
void zendnn_grp_embedding_bag_impl(std::vector <memory> &z_input,
                                   std::vector <memory> &z_indices, std::vector <memory> &z_offsets,
                                   std::vector <int32_t> &z_scale_grad_by_freq, std::vector <algorithm> &z_modes,
                                   std::vector <int32_t> &z_sparse, std::vector <memory> &z_per_sample_weights_opt,
                                   std::vector <int32_t> &z_per_sample_weights_defined,
                                   std::vector <int32_t> &z_include_last_offset,
                                   std::vector <int32_t> &z_padding_idx,
                                   std::vector <memory> &z_destination, const char *plugin_op, int thread_qty,
                                   const bool &scale_bias_last, const int &cat_dim=-1, const int &mlp_pos=-1,
                                   const int &output_stride=-1) {

    zendnnEnv zenEnvObj = readEnv();
    unsigned int eb_thread_qty = zenEnvObj.omp_num_threads;
    int num_tables = z_input.size();
    std::string thread_type;
    int batch_size      = z_destination[0].get_desc().dims()[0];
    int embedding_dim   = z_input[0].get_desc().dims()[1];
    bool cat_output     = (z_destination.size()==1);

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

                zendnn_embedding_bag_exec<IN_TYPE, OUT_TYPE>(
                    z_input[threadOffset], z_indices[threadOffset], z_offsets[threadOffset],
                    z_scale_grad_by_freq[threadOffset], z_modes[threadOffset],
                    z_sparse[threadOffset], z_per_sample_weights_opt[threadOffset],
                    z_per_sample_weights_defined[threadOffset],
                    z_include_last_offset[threadOffset],
                    z_padding_idx[threadOffset],
                    (cat_output)? z_destination[0]:z_destination[threadOffset], plugin_op,
                    inner_threads, scale_bias_last, cat_dim, mlp_pos, output_stride, threadOffset,
                    num_tables);
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
            zendnn_embedding_bag_exec<IN_TYPE, OUT_TYPE>(
                z_input[threadOffset], z_indices[threadOffset], z_offsets[threadOffset],
                z_scale_grad_by_freq[threadOffset], z_modes[threadOffset],
                z_sparse[threadOffset], z_per_sample_weights_opt[threadOffset],
                z_per_sample_weights_defined[threadOffset],
                z_include_last_offset[threadOffset],
                z_padding_idx[threadOffset],
                (cat_output)? z_destination[0]:z_destination[threadOffset], plugin_op,
                inner_threads, scale_bias_last, cat_dim, mlp_pos, output_stride, threadOffset,
                num_tables);
        }
    }

    else if (zenEnvObj.zenEBThreadAlgo==zenEBThreadType::TABLE_THREADED) {
        thread_type="table_threaded";
        unsigned int loopCount = (num_tables%eb_thread_qty)==0 ?
                                 num_tables/eb_thread_qty : ((num_tables/eb_thread_qty)+1);

#if ZENDNN_PARALLEL_FOR_TT
        zendnn_parallel_for(0, num_tables, 1, [&](int start_idx, int end_idx) {
            for (int i = start_idx; i < end_idx; i++) {
                zendnn_embedding_bag_exec<IN_TYPE, OUT_TYPE>(
                    z_input[i], z_indices[i], z_offsets[i],
                    z_scale_grad_by_freq[i], z_modes[i],
                    z_sparse[i], z_per_sample_weights_opt[i],z_per_sample_weights_defined[i],
                    z_include_last_offset[i],
                    z_padding_idx[i], (cat_output)? z_destination[0]:z_destination[i], plugin_op,
                    1, scale_bias_last,
                    cat_dim, mlp_pos, output_stride, i, num_tables);
            }
        });
#else
        #pragma omp parallel num_threads(eb_thread_qty)
        {
            for (int i=0; i<loopCount; i++) {
                int threadOffset = omp_get_thread_num()+ (i*eb_thread_qty);
                if (threadOffset >= num_tables) {
                    break;
                }
                zendnn_embedding_bag_exec<IN_TYPE, OUT_TYPE>(
                    z_input[threadOffset], z_indices[threadOffset], z_offsets[threadOffset],
                    z_scale_grad_by_freq[threadOffset], z_modes[threadOffset],
                    z_sparse[threadOffset], z_per_sample_weights_opt[threadOffset],
                    z_per_sample_weights_defined[threadOffset],
                    z_include_last_offset[threadOffset],
                    z_padding_idx[threadOffset],
                    (cat_output)? z_destination[0]:z_destination[threadOffset], plugin_op, 1,
                    scale_bias_last, cat_dim, mlp_pos, output_stride, threadOffset, num_tables);
            }
        }
#endif
    }

    else {
        thread_type="batch_threaded";
        for (int i = 0; i < num_tables; i++) {
            zendnn_embedding_bag_exec<IN_TYPE, OUT_TYPE>(
                z_input[i], z_indices[i], z_offsets[i],
                z_scale_grad_by_freq[i], z_modes[i],
                z_sparse[i], z_per_sample_weights_opt[i],z_per_sample_weights_defined[i],
                z_include_last_offset[i],
                z_padding_idx[i], (cat_output)? z_destination[0]:z_destination[i], plugin_op,
                eb_thread_qty, scale_bias_last,
                cat_dim, mlp_pos, output_stride, i, num_tables);
        }

    }

    double duration_ms = impl::get_msec() - start_ms;

    zendnnVerbose(ZENDNN_PROFLOG, "zendnn_custom_op_execute,cpu,plugin_op:",
                  plugin_op, ",","num_ops:",num_tables,",","dims:batch_size=",batch_size,",",
                  "alg:",thread_type,",",duration_ms,",ms");
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
        memory &z_destination, const char *plugin_op, int thread_qty,
        const bool &scale_bias_last) {

    zendnnEnv zenEnvObj = readEnv();
    unsigned int eb_thread_qty = zenEnvObj.omp_num_threads;

    if ((z_input.get_desc().data_type()==impl::data_type::f32) &&
            (z_destination.get_desc().data_type()==impl::data_type::f32)) {

        zendnn_embedding_bag_exec<float, float>(
            z_input, z_indices, z_offsets, static_cast<int32_t>(z_scale_grad_by_freq),
            z_mode,
            static_cast<int32_t>(z_sparse), z_per_sample_weights_opt,
            static_cast<int32_t>(z_per_sample_weights_defined),
            static_cast<int32_t>(z_include_last_offset),
            z_padding_idx, z_destination, plugin_op, eb_thread_qty,
            scale_bias_last);
    }
    else if ((z_input.get_desc().data_type()==impl::data_type::bf16) &&
             (z_destination.get_desc().data_type()==impl::data_type::f32)) {

        zendnn_embedding_bag_exec<uint16_t, float>(
            z_input, z_indices, z_offsets, static_cast<int32_t>(z_scale_grad_by_freq),
            z_mode,
            static_cast<int32_t>(z_sparse), z_per_sample_weights_opt,
            static_cast<int32_t>(z_per_sample_weights_defined),
            static_cast<int32_t>(z_include_last_offset),
            z_padding_idx, z_destination, plugin_op, eb_thread_qty,
            scale_bias_last);
    }
    else if ((z_input.get_desc().data_type()==impl::data_type::bf16) &&
             (z_destination.get_desc().data_type()==impl::data_type::bf16)) {

        zendnn_embedding_bag_exec<uint16_t, uint16_t>(
            z_input, z_indices, z_offsets, static_cast<int32_t>(z_scale_grad_by_freq),
            z_mode,
            static_cast<int32_t>(z_sparse), z_per_sample_weights_opt,
            static_cast<int32_t>(z_per_sample_weights_defined),
            static_cast<int32_t>(z_include_last_offset),
            z_padding_idx, z_destination, plugin_op, eb_thread_qty,
            scale_bias_last);
    }
    else if ((z_input.get_desc().data_type()==impl::data_type::s4) &&
             (z_destination.get_desc().data_type()==impl::data_type::bf16)) {

        zendnn_embedding_bag_exec<uint8_t, uint16_t>(
            z_input, z_indices, z_offsets, static_cast<int32_t>(z_scale_grad_by_freq),
            z_mode,
            static_cast<int32_t>(z_sparse), z_per_sample_weights_opt,
            static_cast<int32_t>(z_per_sample_weights_defined),
            static_cast<int32_t>(z_include_last_offset),
            z_padding_idx, z_destination, plugin_op, eb_thread_qty,
            scale_bias_last);
    }
    else if ((z_input.get_desc().data_type()==impl::data_type::s4) &&
             (z_destination.get_desc().data_type()==impl::data_type::f32)) {

        zendnn_embedding_bag_exec<uint8_t, float>(
            z_input, z_indices, z_offsets, static_cast<int32_t>(z_scale_grad_by_freq),
            z_mode,
            static_cast<int32_t>(z_sparse), z_per_sample_weights_opt,
            static_cast<int32_t>(z_per_sample_weights_defined),
            static_cast<int32_t>(z_include_last_offset),
            z_padding_idx, z_destination, plugin_op, eb_thread_qty,
            scale_bias_last);
    }
    else {
        ZENDNN_THROW_ERROR(zendnn_invalid_arguments, "Data Type not Supported");
    }
}

void zendnn_custom_op::zendnn_grp_embedding_bag(std::vector <memory> &z_input,
        std::vector <memory> &z_indices, std::vector <memory> &z_offsets,
        std::vector <int32_t> &z_scale_grad_by_freq, std::vector <algorithm> &z_modes,
        std::vector <int32_t> &z_sparse, std::vector <memory> &z_per_sample_weights_opt,
        std::vector <int32_t> &z_per_sample_weights_defined,
        std::vector <int32_t> &z_include_last_offset,
        std::vector <int32_t> &z_padding_idx,
        std::vector <memory> &z_destination, const char *plugin_op, const int &cat_dim,
        const int &mlp_pos, const int &output_stride, int thread_qty,
        const bool &scale_bias_last) {

    if ((z_input[0].get_desc().data_type()==impl::data_type::f32) &&
            (z_destination[0].get_desc().data_type()==impl::data_type::f32)) {

        zendnn_grp_embedding_bag_impl<float, float>(z_input, z_indices, z_offsets,
                z_scale_grad_by_freq, z_modes,
                z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
                z_include_last_offset,
                z_padding_idx, z_destination, plugin_op, thread_qty, scale_bias_last, cat_dim,
                mlp_pos, output_stride);
    }
    else if ((z_input[0].get_desc().data_type()==impl::data_type::bf16) &&
             (z_destination[0].get_desc().data_type()==impl::data_type::f32)) {

        zendnn_grp_embedding_bag_impl<uint16_t, float>(z_input, z_indices, z_offsets,
                z_scale_grad_by_freq, z_modes,
                z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
                z_include_last_offset,
                z_padding_idx, z_destination, plugin_op, thread_qty, scale_bias_last, cat_dim,
                mlp_pos, output_stride);
    }
    else if ((z_input[0].get_desc().data_type()==impl::data_type::bf16) &&
             (z_destination[0].get_desc().data_type()==impl::data_type::bf16)) {

        zendnn_grp_embedding_bag_impl<uint16_t, uint16_t>(z_input, z_indices, z_offsets,
                z_scale_grad_by_freq, z_modes,
                z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
                z_include_last_offset,
                z_padding_idx, z_destination, plugin_op, thread_qty, scale_bias_last, cat_dim,
                mlp_pos, output_stride);
    }
    else if ((z_input[0].get_desc().data_type()==impl::data_type::s4) &&
             (z_destination[0].get_desc().data_type()==impl::data_type::bf16)) {

        zendnn_grp_embedding_bag_impl<uint8_t, uint16_t>(z_input, z_indices, z_offsets,
                z_scale_grad_by_freq, z_modes,
                z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
                z_include_last_offset,
                z_padding_idx, z_destination, plugin_op, thread_qty, scale_bias_last, cat_dim,
                mlp_pos, output_stride);
    }
    else if ((z_input[0].get_desc().data_type()==impl::data_type::s4) &&
             (z_destination[0].get_desc().data_type()==impl::data_type::f32)) {

        zendnn_grp_embedding_bag_impl<uint8_t, float>(z_input, z_indices, z_offsets,
                z_scale_grad_by_freq, z_modes,
                z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
                z_include_last_offset,
                z_padding_idx, z_destination, plugin_op, thread_qty, scale_bias_last, cat_dim,
                mlp_pos, output_stride);
    }
    else {
        ZENDNN_THROW_ERROR(zendnn_invalid_arguments, "Data Type not Supported");
    }
}

//API call to perform just embedding lookup where each input index corresponds to single embedding.

void zendnn_custom_op::zendnn_embedding(const memory &z_input,
                                        const memory &z_indices,
                                        const int32_t &z_padding_idx, const bool &z_scale_grad_by_freq,
                                        const bool &z_sparse,
                                        memory &z_destination, const char *plugin_op, int thread_qty,
                                        const bool &scale_bias_last) {

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

    if ((z_input.get_desc().data_type()==impl::data_type::f32) &&
            (z_destination.get_desc().data_type()==impl::data_type::f32)) {

        zendnn_embedding_bag_exec<float, float>(
            z_input, z_indices, z_offsets, static_cast<int32_t>(z_scale_grad_by_freq),
            z_mode,
            static_cast<int32_t>(z_sparse), z_per_sample_weights_opt,
            static_cast<int32_t>(z_per_sample_weights_defined),
            static_cast<int32_t>(z_include_last_offset),
            z_padding_idx, z_destination, plugin_op, num_thread, scale_bias_last);
    }
    else if ((z_input.get_desc().data_type()==impl::data_type::bf16) &&
             (z_destination.get_desc().data_type()==impl::data_type::f32)) {

        zendnn_embedding_bag_exec<uint16_t, float>(
            z_input, z_indices, z_offsets, static_cast<int32_t>(z_scale_grad_by_freq),
            z_mode,
            static_cast<int32_t>(z_sparse), z_per_sample_weights_opt,
            static_cast<int32_t>(z_per_sample_weights_defined),
            static_cast<int32_t>(z_include_last_offset),
            z_padding_idx, z_destination, plugin_op, num_thread, scale_bias_last);
    }
    else if ((z_input.get_desc().data_type()==impl::data_type::bf16) &&
             (z_destination.get_desc().data_type()==impl::data_type::bf16)) {

        zendnn_embedding_bag_exec<uint16_t, uint16_t>(
            z_input, z_indices, z_offsets, static_cast<int32_t>(z_scale_grad_by_freq),
            z_mode,
            static_cast<int32_t>(z_sparse), z_per_sample_weights_opt,
            static_cast<int32_t>(z_per_sample_weights_defined),
            static_cast<int32_t>(z_include_last_offset),
            z_padding_idx, z_destination, plugin_op, num_thread, scale_bias_last);
    }
    else if ((z_input.get_desc().data_type()==impl::data_type::s4) &&
             (z_destination.get_desc().data_type()==impl::data_type::bf16)) {

        zendnn_embedding_bag_exec<uint8_t, uint16_t>(
            z_input, z_indices, z_offsets, static_cast<int32_t>(z_scale_grad_by_freq),
            z_mode,
            static_cast<int32_t>(z_sparse), z_per_sample_weights_opt,
            static_cast<int32_t>(z_per_sample_weights_defined),
            static_cast<int32_t>(z_include_last_offset),
            z_padding_idx, z_destination, plugin_op, num_thread, scale_bias_last);
    }
    else if ((z_input.get_desc().data_type()==impl::data_type::s4) &&
             (z_destination.get_desc().data_type()==impl::data_type::f32)) {

        zendnn_embedding_bag_exec<uint8_t, float>(
            z_input, z_indices, z_offsets, static_cast<int32_t>(z_scale_grad_by_freq),
            z_mode,
            static_cast<int32_t>(z_sparse), z_per_sample_weights_opt,
            static_cast<int32_t>(z_per_sample_weights_defined),
            static_cast<int32_t>(z_include_last_offset),
            z_padding_idx, z_destination, plugin_op, num_thread, scale_bias_last);
    }
    else {
        ZENDNN_THROW_ERROR(zendnn_invalid_arguments, "Data Type not Supported");
    }
}

void zendnn_custom_op::zendnn_grp_embedding(std::vector <memory> &z_input,
        std::vector <memory> &z_indices,
        std::vector <int32_t> &z_padding_idx,
        std::vector <int32_t> &z_scale_grad_by_freq,
        std::vector <int32_t> &z_sparse,
        std::vector <memory> &z_destination, const char *plugin_op, int thread_qty,
        const bool &scale_bias_last) {

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

    if ((z_input[0].get_desc().data_type()==impl::data_type::f32) &&
            (z_destination[0].get_desc().data_type()==impl::data_type::f32)) {

        zendnn_grp_embedding_bag_impl<float, float>(
            z_input, z_indices, z_offsets, z_scale_grad_by_freq, z_modes,
            z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
            z_include_last_offset, z_padding_idx, z_destination, plugin_op, thread_qty,
            scale_bias_last);
    }
    else if ((z_input[0].get_desc().data_type()==impl::data_type::bf16) &&
             (z_destination[0].get_desc().data_type()==impl::data_type::f32)) {

        zendnn_grp_embedding_bag_impl<uint16_t, float>(
            z_input, z_indices, z_offsets, z_scale_grad_by_freq, z_modes,
            z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
            z_include_last_offset, z_padding_idx, z_destination, plugin_op, thread_qty,
            scale_bias_last);
    }
    else if ((z_input[0].get_desc().data_type()==impl::data_type::bf16) &&
             (z_destination[0].get_desc().data_type()==impl::data_type::bf16)) {

        zendnn_grp_embedding_bag_impl<uint16_t, uint16_t>(
            z_input, z_indices, z_offsets, z_scale_grad_by_freq, z_modes,
            z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
            z_include_last_offset, z_padding_idx, z_destination, plugin_op, thread_qty,
            scale_bias_last);
    }
    else if ((z_input[0].get_desc().data_type()==impl::data_type::s4) &&
             (z_destination[0].get_desc().data_type()==impl::data_type::bf16)) {

        zendnn_grp_embedding_bag_impl<uint8_t, uint16_t>(
            z_input, z_indices, z_offsets, z_scale_grad_by_freq, z_modes,
            z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
            z_include_last_offset, z_padding_idx, z_destination, plugin_op, thread_qty,
            scale_bias_last);
    }
    else if ((z_input[0].get_desc().data_type()==impl::data_type::s4) &&
             (z_destination[0].get_desc().data_type()==impl::data_type::f32)) {

        zendnn_grp_embedding_bag_impl<uint8_t, float>(
            z_input, z_indices, z_offsets, z_scale_grad_by_freq, z_modes,
            z_sparse, z_per_sample_weights_opt, z_per_sample_weights_defined,
            z_include_last_offset, z_padding_idx, z_destination, plugin_op, thread_qty,
            scale_bias_last);
    }
    else {
        ZENDNN_THROW_ERROR(zendnn_invalid_arguments, "Data Type not Supported");
    }

}
}//ZenDNN

