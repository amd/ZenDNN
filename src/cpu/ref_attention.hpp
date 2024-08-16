/*******************************************************************************
* Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef CPU_ATTENTION_HPP
#define CPU_ATTENTION_HPP

#include <iostream>
#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/primitive.hpp"
#include "common/engine.hpp"
#include "common/stream.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_attention_pd.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "common/zendnn_thread.hpp"

#include <blis.h>
#include "zendnn.hpp"
#include "common/zendnn_private.hpp"

//#define DEBUG_ATTN

namespace zendnn {
namespace impl {
namespace utils {
void fill_offset(std::vector<unsigned long> &offsets,
                        unsigned int offset_index,
                        unsigned int curr_offset,
                        int64_t const dims1[],
                        int64_t const dims2[],
                        unsigned int dims_len,
                        unsigned int dims_index,
                        unsigned int mat_size) {

    if (dims_len == 0) {
        return;
    }
    if (dims_index == dims_len - 1) {
        offsets[offset_index] = curr_offset + mat_size;
        offset_index++;
        if (dims1[dims_index] == dims2[dims_index]) {
            for (int i = 1; i < dims1[dims_index]; i++) {
                offsets[offset_index] = offsets[offset_index - 1] + mat_size;
                offset_index++;
            }
        }
        else {
            if (dims1[dims_index] == 1) {
                for (int i = 1; i < dims2[dims_index]; i++) {
                    offsets[offset_index] = offsets[offset_index - 1];
                    offset_index++;
                }
            }
        }
        return;
    }
    unsigned int count = 1;
    for (int j = dims_index + 1; j < dims_len; j++) {
        count = count * dims2[j];
    }
    if (dims1[dims_index] == dims2[dims_index]) {
        int current_offset = curr_offset;
        for (int i = 0; i < dims1[dims_index]; i++) {
            fill_offset(offsets, offset_index, current_offset, dims1, dims2, dims_len,
                        dims_index + 1, mat_size);
            offset_index += count;
            current_offset = offsets[offset_index - 1];
        }
    }
    else {
        if (dims1[dims_index] == 1) {
            for (int i = 0; i < dims2[dims_index]; i++) {
                fill_offset(offsets, offset_index, curr_offset, dims1, dims2, dims_len,
                            dims_index + 1, mat_size);
                offset_index += count;
            }
        }
    }
    return;
}

void calculate_offsets(std::vector<unsigned long> &offsets,
                              int64_t const dims1[],
                              int64_t const dims2[],
                              unsigned int dims_len,
                              unsigned long mat_dim1,
                              unsigned long mat_dim2) {
    fill_offset(offsets, 0, -1*mat_dim1*mat_dim2, dims1, dims2, dims_len, 0,
                mat_dim1*mat_dim2);
    return;
}
} //namespace utils

namespace cpu {
namespace attention {

void zenAttention_Matmul(const float *a,
            memory_desc_wrapper a_mdw,
            const float *b,
            memory_desc_wrapper b_mdw,
            float scale,
            const float *c,
            memory_desc_wrapper c_mdw,
            float *o,
            memory_desc_wrapper o_mdw) {

    zendnn::impl::cpu::matmul::matmul_helper_t helper(a_mdw, b_mdw, o_mdw);
    const int ndims = o_mdw.ndims();
    const int batch_ndims = ndims - 2;
    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const dim_t batch = helper.batch();

    std::vector<unsigned long> o_off, a_off, b_off;

    a_off.resize(batch);
    b_off.resize(batch);
    o_off.resize(batch);

    zendnn::impl::utils::calculate_offsets(a_off, (int64_t *)a_mdw.dims(),
                                           (int64_t *)a_mdw.dims(),
                                           a_mdw.ndims() - 2,
                                           M,
                                           K);
    zendnn::impl::utils::calculate_offsets(b_off, (int64_t *)b_mdw.dims(),
                                           (int64_t *)b_mdw.dims(),
                                           b_mdw.ndims() - 2,
                                           K,
                                           N);
    zendnn::impl::utils::calculate_offsets(o_off, (int64_t *)o_mdw.dims(),
                                           (int64_t *)o_mdw.dims(),
                                           o_mdw.ndims() - 2,
                                           M,
                                           N);
    unsigned long *a_offsets = a_off.data();
    unsigned long *b_offsets = b_off.data();
    unsigned long *o_offsets = o_off.data();

    const bool Layout = true; // CblasRowMajor

    const auto &a_strides = &a_mdw.blocking_desc().strides[a_mdw.ndims() - 2];
    const auto &b_strides = &b_mdw.blocking_desc().strides[b_mdw.ndims() - 2];
    const auto &o_bd = o_mdw.blocking_desc();
    // In case of normal matrices, the stride of the last dimension will always be 1,
    // as the elements are contiguous. However in case of transposed matrix, the
    // stride of the last dimension will be greater than 1.
    const char *transA
        = a_strides[1] == 1 ? "N" : "T";
    const char *transB
        = b_strides[1] == 1 ? "N" : "T";

    const dim_t M_s32 = (dim_t)M;
    const dim_t N_s32 = (dim_t)N;
    const dim_t K_s32 = (dim_t)K;

    const dim_t lda = (dim_t)a_strides[*transA == 'N' ? 0 : 1];
    const dim_t ldb = (dim_t)b_strides[*transB == 'N' ? 0 : 1];
    const dim_t ldc = (dim_t)o_bd.strides[o_mdw.ndims() - 2];

    //MatMul with Bias
#ifdef DEBUG_ATTN
    zendnnInfo(ZENDNN_CORELOG,"[Custom] M: ", M);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] N: ", N);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] K: ", K);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] lda: ", lda);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] ldb: ", ldb);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] ldc: ", ldc);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] batch: ", batch);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] transA: ", transA);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] transB: ", transB);
#endif

#if 1
    zenMatMulWithBias(Layout, strcmp(transA, "N"), strcmp(transB, "N"),
                      batch, a_offsets, b_offsets, o_offsets,
                      M, K, N,
                      scale,//alpha
                      (float *)a, lda,
                      (float *)b, ldb,
                      (float *)c,
                      0.0f,//beta
                      (float *)o, ldc);
#else
    zenMatMul(Layout, strcmp(transA, "N"), strcmp(transB, "N"),
              batch, a_offsets, b_offsets, o_offsets,
              M, K, N,
              scale,//alpha
              (float *)a, lda,
              (float *)b, ldb,
              (float *)c,
              (bool)0,
              (int)0,
              0.0f,//beta
              (float *)o, ldc);
#endif
    return;
}

/* TODO: Is transpose really necessary here?? Please check for optimizing!! */
void zenAttention_Transpose(float *src,
                            zendnn::memory::desc src_md,
                            float *dst,
                            zendnn::memory::desc dst_md,
                            std::vector<int> perm) {
    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    auto data_dims = src_md.dims();
    auto ndata_dims = src_md.dims().size();
    zendnn::memory::dims transposed_dims(ndata_dims, 0);
    zendnn::memory::dims strides(ndata_dims, 0);
    zendnn::memory::dim total_stride = 1;
    for (int i = (int)ndata_dims - 1 ; i >= 0; i--) {
        transposed_dims[i] = data_dims[perm[i]];
        strides[perm[i]] = total_stride;
        total_stride *= data_dims[perm[i]];
    }

    zendnn::memory src_mem({data_dims, zendnn::memory::data_type::f32, zendnn::memory::format_tag::abcd}, eng, (void*)src);
    zendnn::memory int_mem({data_dims, zendnn::memory::data_type::f32, strides}, eng, (void*)dst);

    zendnn::reorder(src_mem, int_mem).execute(engine_stream, src_mem, int_mem);
}

void zenAttention_TransformAddMask(float *qkbuff,
                                   zendnn::memory::desc qk_md,
                                   const float *mask,
                                   zendnn::memory::desc mask_md,
                                   float *scratchpad) {
    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    zendnn::memory qk_mem(qk_md, eng, (void*)qkbuff);
    zendnn::memory mask_mem(mask_md, eng, (void*)mask);
    zendnn::memory intermediate_mem(mask_md, eng, (void*)scratchpad);

    auto linear_d = zendnn::eltwise_forward::desc(zendnn::prop_kind::forward_inference,
                                                  zendnn::algorithm::eltwise_linear,
                                                  mask_md,
                                                  10000.0f, -10000.0f);

    auto linear_pd = zendnn::eltwise_forward::primitive_desc(linear_d, eng);

    net_args.push_back({{ZENDNN_ARG_SRC_0, mask_mem},
                        {ZENDNN_ARG_DST, intermediate_mem}});

    net.push_back(zendnn::eltwise_forward(linear_pd));

    auto binary_d = zendnn::binary::desc(zendnn::algorithm::binary_add,
                                         qk_md,
                                         intermediate_mem.get_desc(),
                                         qk_md
                                         );

    auto binary_pd = zendnn::binary::primitive_desc(binary_d, eng);

    net_args.push_back({{ZENDNN_ARG_SRC_0, qk_mem},
                        {ZENDNN_ARG_SRC_1, intermediate_mem},
                        {ZENDNN_ARG_DST, qk_mem}});

    net.push_back(zendnn::binary(binary_pd));

    for (size_t i=0; i<net.size(); ++i) {
        net.at(i).execute(engine_stream, net_args.at(i));
    }
    zendnnInfo(ZENDNN_CORELOG,"[Custom] zenAttention_TransformAddMask() ");
}

void zenAttention_Softmax(const float *src,
                          zendnn::memory::desc src_md,
                          int axis) {
    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    zendnn::memory src_mem(src_md, eng, (void*)src);

    auto softmax_desc = zendnn::softmax_forward::desc(zendnn::prop_kind::forward_inference,
                                                      src_md,
                                                      (int) axis);
    auto softmax_pd = zendnn::softmax_forward::primitive_desc(softmax_desc, eng);

    net_args.push_back({{ZENDNN_ARG_SRC, src_mem},
                        {ZENDNN_ARG_DST, src_mem}});

    net.push_back(zendnn::softmax_forward(softmax_pd));

    for (size_t i = 0; i < net.size(); ++i) {
        net.at(i).execute(engine_stream, net_args.at(i));
    }
    zendnnInfo(ZENDNN_CORELOG,"[Custom] zenAttention_Softmax() ");
}

} // namespace attentiom

/* add new primitive */
template <impl::data_type_t data_type>
struct ref_attention_t : public primitive_t {
    struct pd_t : public cpu_attention_pd_t {
        using cpu_attention_pd_t::cpu_attention_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_attention_t);

        status_t init(engine_t *engine) {
            if (!platform::has_data_type_support(data_type)) {
                return status::unimplemented;
            }

            init_scratchpad();
            return status::success;
        }
        private:
            void init_scratchpad() {
                auto scratchpad = scratchpad_registry().registrar();
                zendnnInfo(ZENDNN_CORELOG, "init_scratchpad()");

                memory_desc_wrapper dMD(this->dst_md(ZENDNN_ARG_DST));
                auto B = dMD.dims()[0]; //B
                auto S = dMD.dims()[1]; //S
                auto N = this->desc()->num_heads;   //N
                auto H = (dim_t)dMD.dims()[2]/N;    //Od
                zendnnInfo(ZENDNN_CORELOG, "init_scratchpad() B : ", B);
                zendnnInfo(ZENDNN_CORELOG, "init_scratchpad() S : ", S);
                zendnnInfo(ZENDNN_CORELOG, "init_scratchpad() N : ", N);
                zendnnInfo(ZENDNN_CORELOG, "init_scratchpad() H : ", H);

                auto scratchpad_size = (4*B*N*S*H + B*N*S*S);

                zendnnInfo(ZENDNN_CORELOG, "init_scratchpad() scratchpad_size : ", scratchpad_size);

                scratchpad.book(memory_tracking::names::key_attention,
                    scratchpad_size, types::data_type_size(zendnn_f32));
            }
    };
    // constructor using pd_t
    ref_attention_t(const pd_t *apd) : primitive_t(apd) {}

    // init() override from primitive_t
    status_t init(engine_t *engine) override {
        return status::success;
    }

    // exec() override from primitive_t
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

  private:
    const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }

    status_t execute_ref(const exec_ctx_t &ctx) const;
};

template<data_type_t data_type>
status_t
ref_attention_t<data_type>::execute_ref(const exec_ctx_t &ctx) const {
    status_t status = status::success;

    // get algorithm params
    auto alg = pd()->desc()->alg_kind;
    auto scale = pd()->desc()->scale;
    auto num_heads = pd()->desc()->num_heads;
    auto num_threads = pd()->desc()->num_threads;

    // get the tensors
    auto query   = CTX_IN_MEM(const float *, ZENDNN_ARG_SRC_0);
    auto key   = CTX_IN_MEM(const float *, ZENDNN_ARG_SRC_1);
    auto value   = CTX_IN_MEM(const float *, ZENDNN_ARG_SRC_2);
    auto weights_query   = CTX_IN_MEM(const float *, ZENDNN_ARG_WEIGHTS_0);
    auto weights_key   = CTX_IN_MEM(const float *, ZENDNN_ARG_WEIGHTS_1);
    auto weights_value   = CTX_IN_MEM(const float *, ZENDNN_ARG_WEIGHTS_2);
    auto bias_query   = CTX_IN_MEM(const float *, ZENDNN_ARG_BIAS_0);
    auto bias_key   = CTX_IN_MEM(const float *, ZENDNN_ARG_BIAS_1);
    auto bias_value   = CTX_IN_MEM(const float *, ZENDNN_ARG_BIAS_2);
    auto mask  = CTX_IN_MEM(const float *, ZENDNN_ARG_MASK);
    auto dst     = CTX_OUT_MEM(float *, ZENDNN_ARG_DST);

    const auto scratchpad = ctx.get_scratchpad_grantor();
    auto scratchpad_buf_base = scratchpad.template get<float>(
                               memory_tracking::names::key_attention);

    // get memory descriptors
    memory_desc_wrapper query_mdw(pd()->src_md(ZENDNN_ARG_SRC_0));
    memory_desc_wrapper key_mdw(pd()->src_md(ZENDNN_ARG_SRC_1));
    memory_desc_wrapper value_mdw(pd()->src_md(ZENDNN_ARG_SRC_2));
    memory_desc_wrapper weights_query_mdw(pd()->src_md(ZENDNN_ARG_WEIGHTS_0));
    memory_desc_wrapper weights_key_mdw(pd()->src_md(ZENDNN_ARG_WEIGHTS_1));
    memory_desc_wrapper weights_value_mdw(pd()->src_md(ZENDNN_ARG_WEIGHTS_2));
    memory_desc_wrapper bias_query_mdw(pd()->src_md(ZENDNN_ARG_BIAS_0));
    memory_desc_wrapper bias_key_mdw(pd()->src_md(ZENDNN_ARG_BIAS_1));
    memory_desc_wrapper bias_value_mdw(pd()->src_md(ZENDNN_ARG_BIAS_2));
    memory_desc_wrapper mask_mdw(pd()->src_md(ZENDNN_ARG_MASK));
    memory_desc_wrapper dst_mdw(pd()->dst_md(ZENDNN_ARG_DST));

#ifdef DEBUG_ATTN
    if(pd()->attr()->scratchpad_mode_ == scratchpad_mode::user) {
        zendnnInfo(ZENDNN_CORELOG, "scratchpad_mode::user");
    } else {
        zendnnInfo(ZENDNN_CORELOG, "scratchpad_mode::library");
    }
#endif
    engine_t *engine = ctx.stream()->engine();

    // initialize output to zero
    std::fill(dst, (dst + dst_mdw.nelems()), 0);

    /* Intermediate Q-buffer memory setup */
    memory_desc_t Qbuff_md;
    std::vector<dim_t> qDims = {query_mdw.dims()[0], query_mdw.dims()[1], weights_query_mdw.dims()[1]};
    zendnn::impl::dims_t qStrides{qDims[1]*qDims[2], qDims[2], 1};
    zendnn_memory_desc_init_by_strides(&Qbuff_md, qDims.size(), qDims.data(), zendnn::impl::data_type::f32, qStrides);
    memory_desc_wrapper Qbuff_mdw(Qbuff_md);
    auto qbuff_size = utils::array_product(Qbuff_mdw.dims(), Qbuff_mdw.ndims());
    auto scp_qBuff = scratchpad_buf_base;
    auto temp_buff_size = qbuff_size;
    auto scp_qtBuff = scratchpad_buf_base + (temp_buff_size);
#ifdef DEBUG_ATTN
    zendnnInfo(ZENDNN_CORELOG,"[Custom] scratchpad_buf_base : ", scratchpad_buf_base);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] temp_buff_size : ", temp_buff_size);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] scp_qtBuff : ", scp_qtBuff);
#endif
    zendnn::impl::cpu::attention::zenAttention_Matmul(query, query_mdw,
                                                      weights_query, weights_query_mdw,
                                                      1.0f,
                                                      bias_query, bias_query_mdw,
                                                      scp_qBuff, Qbuff_mdw);
    int i=0, id=0;
    //Qb_reshape
    std::vector<dim_t> rQDims = {Qbuff_mdw.dims()[0], Qbuff_mdw.dims()[1], (dim_t)num_heads, (dim_t)Qbuff_mdw.dims()[2]/num_heads};
    memory_desc_t rQ_md;
    zendnn_status_t reshapeSts = zendnn_memory_desc_reshape(&rQ_md, (const zendnn_memory_desc_t*)&Qbuff_md, Qbuff_mdw.ndims()+1, (const zendnn_dim_t *)rQDims.data());
    if(reshapeSts != zendnn_success)
        zendnnInfo(ZENDNN_CORELOG,"[Custom] reshape unsuccessfull. Re-check!!");
    memory_desc_wrapper rQbuff_mdw(rQ_md);
    //Qb_transpose
    std::vector<int> Qperm{0, 2, 1, 3};
    memory_desc_t rtQ_md;
    std::vector<dim_t> rtQDims;
    for(int i=0; i<rQbuff_mdw.ndims(); i++) {
        rtQDims.push_back(rQbuff_mdw.dims()[Qperm[i]]);
    }
    zendnn::impl::dims_t rtQStrides{rtQDims[1]*rtQDims[2]*rtQDims[3], rtQDims[2]*rtQDims[3], rtQDims[3], 1};
    zendnn_memory_desc_init_by_strides(&rtQ_md, rtQDims.size(), rtQDims.data(), zendnn::impl::data_type::f32, rtQStrides);
    memory_desc_wrapper rtQbuff_mdw(rtQ_md);
    zendnn::impl::cpu::attention::zenAttention_Transpose(scp_qBuff, rQ_md, scp_qtBuff, rtQ_md, Qperm);
    //Qb_reshape //Qb_transpose

    /* Intermediate K-buffer memory setup */
    memory_desc_t Kbuff_md;
    std::vector<dim_t> kDims = {key_mdw.dims()[0], key_mdw.dims()[1], weights_key_mdw.dims()[1]};
    zendnn::impl::dims_t kStrides{kDims[1]*kDims[2], kDims[2], 1};
    zendnn_memory_desc_init_by_strides(&Kbuff_md, kDims.size(), kDims.data(), zendnn::impl::data_type::f32, kStrides);
    memory_desc_wrapper Kbuff_mdw(Kbuff_md);
    //scratchpad_buf_base is BSNH size
    auto kbuff_size = utils::array_product(Kbuff_mdw.dims(), Kbuff_mdw.ndims());
    auto scp_kBuff = scratchpad_buf_base;
    auto scp_ktBuff = scratchpad_buf_base + (temp_buff_size + qbuff_size);
#ifdef DEBUG_ATTN
    zendnnInfo(ZENDNN_CORELOG,"[Custom] scp_kBuff : ", scp_kBuff);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] temp_buff_size + qbuff_size : ", temp_buff_size + qbuff_size);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] scp_ktBuff : ", scp_ktBuff);
#endif
    zendnn::impl::cpu::attention::zenAttention_Matmul(key, key_mdw,
                                                      weights_key, weights_key_mdw,
                                                      1.0f,
                                                      bias_key, bias_key_mdw,
                                                      scp_kBuff, Kbuff_mdw);
    //Kb_reshape
    std::vector<dim_t> rKDims = {Kbuff_mdw.dims()[0], Kbuff_mdw.dims()[1], (dim_t)num_heads, (dim_t)Kbuff_mdw.dims()[2]/num_heads};
    memory_desc_t rK_md;
    reshapeSts = zendnn_memory_desc_reshape(&rK_md, (const zendnn_memory_desc_t*)&Kbuff_md, Kbuff_mdw.ndims()+1, (const zendnn_dim_t *)rKDims.data());
    if(reshapeSts != zendnn_success)
        zendnnInfo(ZENDNN_CORELOG,"[Custom] reshape unsuccessfull. Re-check!!: ");
    memory_desc_wrapper rKbuff_mdw(rK_md);
    //Kb_transpose
    std::vector<int> Kperm{0, 2, 3, 1};
    memory_desc_t rtK_md;
    std::vector<dim_t> rtKDims;
    for(int i=0; i<rKbuff_mdw.ndims(); i++) {
        rtKDims.push_back(rKbuff_mdw.dims()[Kperm[i]]);
    }
    zendnn::impl::dims_t rtKStrides{rtKDims[1]*rtKDims[2]*rtKDims[3], rtKDims[2]*rtKDims[3], rtKDims[3], 1};
    zendnn_memory_desc_init_by_strides(&rtK_md, rtKDims.size(), rtKDims.data(), zendnn::impl::data_type::f32, rtKStrides);
    memory_desc_wrapper rtKbuff_mdw(rtK_md);
    zendnn::impl::cpu::attention::zenAttention_Transpose(scp_kBuff, rK_md, scp_ktBuff, rtK_md, Kperm);
    //Kb_reshape //Kb_transpose

    /* Intermediate V-buffer memory setup */
    memory_desc_t Vbuff_md;
    std::vector<dim_t> vDims = {value_mdw.dims()[0], value_mdw.dims()[1], weights_value_mdw.dims()[1]};
    zendnn::impl::dims_t vStrides{vDims[1]*vDims[2], vDims[2], 1};
    zendnn_memory_desc_init_by_strides(&Vbuff_md, vDims.size(), vDims.data(), zendnn::impl::data_type::f32, vStrides);
    memory_desc_wrapper Vbuff_mdw(Vbuff_md);
    //scratchpad_buf_base is BSNH size
    auto vbuff_size = utils::array_product(Vbuff_mdw.dims(), Vbuff_mdw.ndims());
    auto scp_vBuff = scratchpad_buf_base;
    auto scp_vtBuff = scratchpad_buf_base + (temp_buff_size + qbuff_size + kbuff_size);
#ifdef DEBUG_ATTN
    zendnnInfo(ZENDNN_CORELOG,"[Custom] scp_vBuff : ", scp_vBuff);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] temp_buff_size + qbuff_size + kbuff_size : ", temp_buff_size + qbuff_size + kbuff_size);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] scp_vtBuff : ", scp_vtBuff);
#endif
    zendnn::impl::cpu::attention::zenAttention_Matmul(value, value_mdw,
                                                      weights_value, weights_value_mdw,
                                                      1.0f,
                                                      bias_value, bias_value_mdw,
                                                      scp_vBuff, Vbuff_mdw);
    //Vb_reshape
    std::vector<dim_t> rVDims = {Vbuff_mdw.dims()[0], Vbuff_mdw.dims()[1], (dim_t)num_heads, (dim_t)Vbuff_mdw.dims()[2]/num_heads};
    memory_desc_t rV_md;
    reshapeSts = zendnn_memory_desc_reshape(&rV_md, (const zendnn_memory_desc_t*)&Vbuff_md, Vbuff_mdw.ndims()+1, (const zendnn_dim_t *)rVDims.data());
    if(reshapeSts != zendnn_success)
        zendnnInfo(ZENDNN_CORELOG,"[Custom] reshape unsuccessfull. Re-checV!!: ");
    memory_desc_wrapper rVbuff_mdw(rV_md);
    //Vb_transpose
    std::vector<int> Vperm{0, 2, 1, 3};
    memory_desc_t rtV_md;
    std::vector<dim_t> rtVDims;
    for(int i=0; i<rVbuff_mdw.ndims(); i++) {
        rtVDims.push_back(rVbuff_mdw.dims()[Vperm[i]]);
    }
    zendnn::impl::dims_t rtVStrides{rtVDims[1]*rtVDims[2]*rtVDims[3], rtVDims[2]*rtVDims[3], rtVDims[3], 1};
    zendnn_memory_desc_init_by_strides(&rtV_md, rtVDims.size(), rtVDims.data(), zendnn::impl::data_type::f32, rtVStrides);
    memory_desc_wrapper rtVbuff_mdw(rtV_md);
    zendnn::impl::cpu::attention::zenAttention_Transpose(scp_vBuff, rV_md, scp_vtBuff, rtV_md, Vperm);
    //Vb_reshape //Vb_transpose

    //4DMatmul(QK', 1/N)
    /* Intermediate QK' buffer memory setup */
    memory_desc_t QKbuff_md;
    std::vector<dim_t> qkDims = {rtQbuff_mdw.dims()[0], rtQbuff_mdw.dims()[1], rtQbuff_mdw.dims()[2], rtKbuff_mdw.dims()[3]};
    zendnn::impl::dims_t qkStrides{qkDims[1]*qkDims[2]*qkDims[3], qkDims[2]*qkDims[3], qkDims[3], 1};
    zendnn_memory_desc_init_by_strides(&QKbuff_md, qkDims.size(), qkDims.data(), zendnn::impl::data_type::f32, qkStrides);
    memory_desc_wrapper QKbuff_mdw(QKbuff_md);
    auto scp_qk = scratchpad_buf_base + (temp_buff_size + qbuff_size + kbuff_size + vbuff_size);
#ifdef DEBUG_ATTN
    zendnnInfo(ZENDNN_CORELOG,"[Custom] scratchpad_buf_base : ", scratchpad_buf_base);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] temp_buff_size + qbuff_size + kbuff_size + vbuff_size : ", temp_buff_size + qbuff_size + kbuff_size + vbuff_size);
    zendnnInfo(ZENDNN_CORELOG,"[Custom] scp_qk : ", scp_qk);
#endif
    zendnn::impl::cpu::attention::zenAttention_Matmul(scp_qtBuff, rtQbuff_mdw,
                                                      scp_ktBuff, rtKbuff_mdw,
                                                      scale,
                                                      nullptr, bias_value_mdw,
                                                      scp_qk, QKbuff_mdw);

    /* Mask memory setup */
    zendnn_memory_desc_t mask_md = *(pd()->src_md(ZENDNN_ARG_MASK));
    //mask_reshape
    std::vector<dim_t> maskDims = {mask_mdw.dims()[0], 1, 1, mask_mdw.dims()[1]};
    memory_desc_t rMask_md;
    reshapeSts = zendnn_memory_desc_reshape(&rMask_md, (const zendnn_memory_desc_t*)&mask_md, maskDims.size(), (const zendnn_dim_t *)maskDims.data());
    if(reshapeSts != zendnn_success)
        zendnnInfo(ZENDNN_CORELOG,"[Custom] reshape unsuccessfull. Re-check!!: ");
    /* Transform mask memory for masking AND add to QKBuffer */
    zendnn::impl::cpu::attention::zenAttention_TransformAddMask(scp_qk, QKbuff_md, mask, rMask_md, scratchpad_buf_base);
    /* Applying softmax */
    zendnn::impl::cpu::attention::zenAttention_Softmax(scp_qk, QKbuff_md, 3);

    /* Intermediate QK'V buffer memory setup */
    memory_desc_t QKVbuff_md;
    std::vector<dim_t> qkvDims = {QKbuff_mdw.dims()[0], QKbuff_mdw.dims()[1], QKbuff_mdw.dims()[2], rtVbuff_mdw.dims()[3]};
    zendnn::impl::dims_t qkvStrides{qkvDims[1]*qkvDims[2]*qkvDims[3], qkvDims[2]*qkvDims[3], qkvDims[3], 1};
    zendnn_memory_desc_init_by_strides(&QKVbuff_md, qkvDims.size(), qkvDims.data(), zendnn::impl::data_type::f32, qkvStrides);
    memory_desc_wrapper QKVbuff_mdw(QKVbuff_md);
    zendnn::impl::cpu::attention::zenAttention_Matmul(scp_qk, QKbuff_mdw,
                                                      scp_vtBuff, rtVbuff_mdw,
                                                      1.0f,
                                                      nullptr, bias_value_mdw,
                                                      scratchpad_buf_base, QKVbuff_mdw);
    //Dst traspose
    memory_desc_t tDst_md;
    std::vector<int> dstPerm{0, 2, 1, 3};
    std::vector<dim_t> tDstDims;
    for(int i=0; i<QKVbuff_mdw.ndims(); i++) {
        tDstDims.push_back(QKVbuff_mdw.dims()[dstPerm[i]]);
    }
    zendnn::impl::dims_t tDstStrides{tDstDims[1]*tDstDims[2]*tDstDims[3], tDstDims[2]*tDstDims[3], tDstDims[3], 1};
    zendnn_memory_desc_init_by_strides(&tDst_md, tDstDims.size(), tDstDims.data(), zendnn::impl::data_type::f32, tDstStrides);
    memory_desc_wrapper tDst_mdw(tDst_md);
    zendnn::impl::cpu::attention::zenAttention_Transpose(scratchpad_buf_base, QKVbuff_md, dst, tDst_md, dstPerm);

    //Dst reshape is not reqiured as only the dst memory descriptor changes come-in
    return status;
}

} // namespace cpu
} // namespace impl
} // namespace zendnn

#endif
