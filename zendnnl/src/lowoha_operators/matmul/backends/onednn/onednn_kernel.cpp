/*******************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_operators/matmul/backends/onednn/onednn_kernel.hpp"
#include <cstring>
#include <mutex>
#include <vector>
#include "common/data_types.hpp"
#include "common/hash_object.hpp"
#include <unordered_map>

namespace zendnnl {
namespace lowoha {
namespace matmul {

#if ZENDNNL_DEPENDS_ONEDNN

namespace {
std::unordered_map<Key_matmul, size_t> &get_onednn_hash_values() {
    static std::unordered_map<Key_matmul, size_t> hash_values;
    return hash_values;
}
lru_cache_t<Key_matmul, dnnl::memory> &get_onednn_matmul_weight_cache() {
    static lru_cache_t<Key_matmul, dnnl::memory> matmul_weight_cache;
    return matmul_weight_cache;
}
std::mutex &get_onednn_blocked_weight_mutex() {
    static std::mutex blocked_weight_mutex;
    return blocked_weight_mutex;
}

void hashValue(size_t &h, size_t value) {
    h = zendnnl::common::hash_combine(h, value);
}

void hashDims(size_t &h, const std::vector<int64_t> &dims) {
    hashValue(h, dims.size());
    for (int64_t dim : dims) {
        hashValue(h, static_cast<size_t>(dim));
    }
}

size_t hashWeightLayout(
        const onednn_utils_t::onednn_matmul_params &dnnl_params) {
    size_t h = 0;
    hashValue(h, static_cast<size_t>(dnnl_params.weights.dtype));
    hashDims(h, dnnl_params.weights.dims);
    hashDims(h, dnnl_params.weights.strides);
    return h;
}

Key_matmul make_identity_weight_key(bool transB, int K, int N, int ldb,
        const onednn_utils_t::onednn_matmul_params &dnnl_params) {
    return Key_matmul(transB, K, N, ldb, dnnl_params.weights.buffer,
            static_cast<uint32_t>(matmul_algo_t::onednn_blocked),
            hashWeightLayout(dnnl_params));
}

/**
 * @brief Computes hash value for blocked memory descriptor
 *
 * Creates a hash from the memory descriptor's strides and blocking info
 * to uniquely identify the blocking format.
 *
 * @param mem_desc Memory descriptor to hash
 * @return Hash value representing the blocking format
 */
size_t hashBlockingDesc(const dnnl::memory::desc &mem_desc) {
    size_t hash_value = 0;
    // Mersenne prime number to avoid collisions
    const size_t prime = 31;
    for (const auto stride : mem_desc.get_strides()) {
        hash_value = hash_value * prime + std::hash<int64_t> {}(stride);
    }
    const int inner_nblks = mem_desc.get_inner_nblks();
    hash_value = hash_value * prime + std::hash<int> {}(inner_nblks);
    const auto inner_blks = mem_desc.get_inner_blks();
    const auto inner_idxs = mem_desc.get_inner_idxs();
    for (int i = 0; i < inner_nblks; ++i) {
        hash_value = hash_value * prime + std::hash<int64_t> {}(inner_blks[i]);
        hash_value = hash_value * prime + std::hash<int64_t> {}(inner_idxs[i]);
    }
    return hash_value;
}
} // namespace

void clear_onednn_matmul_weight_cache() {
    std::lock_guard<std::mutex> lock(get_onednn_blocked_weight_mutex());
    get_onednn_hash_values().clear();
    get_onednn_matmul_weight_cache().clear();
}

/**
 * @brief Creates matmul primitive descriptor with blocked weight format
 *
 * Creates memory descriptors for input, weights, output, and bias tensors,
 * then builds a matmul primitive descriptor using "any" format for weights
 * to allow oneDNN to choose optimal blocking.
 *
 * @param dnnl_params OneDNN parameters containing tensor info
 * @param eng OneDNN engine
 * @param matmul_attr Primitive attributes (post-ops, scales, etc.)
 * @return matmul primitive descriptor with optimal weight blocking
 */
dnnl::matmul::primitive_desc create_blocked_matmul_pd(
        onednn_utils_t::onednn_matmul_params &dnnl_params,
        const dnnl::engine &eng, const dnnl::primitive_attr &matmul_attr) {

    dnnl::memory::desc dnnl_input_desc
            = onednn_utils_t::to_dnnl_tensor(dnnl_params.src, eng);

    dnnl_params.weights.format_tag = "any";
    dnnl::memory::desc dnnl_blocked_weight_desc
            = onednn_utils_t::to_dnnl_tensor(dnnl_params.weights, eng);

    dnnl::memory::desc dnnl_output_desc
            = onednn_utils_t::to_dnnl_tensor(dnnl_params.dst, eng);

    if (dnnl_params.bias.buffer != nullptr) {
        dnnl::memory::desc dnnl_bias_desc
                = onednn_utils_t::to_dnnl_tensor(dnnl_params.bias, eng);
        return dnnl::matmul::primitive_desc(eng, dnnl_input_desc,
                dnnl_blocked_weight_desc, dnnl_bias_desc, dnnl_output_desc,
                matmul_attr);
    }
    return dnnl::matmul::primitive_desc(eng, dnnl_input_desc,
            dnnl_blocked_weight_desc, dnnl_output_desc, matmul_attr);
}

void getOrCreateBlockedWeights(bool transA, bool transB, int M, int K, int N,
        int lda, int ldb, onednn_utils_t::onednn_matmul_params &dnnl_params,
        const dnnl::engine &eng, const dnnl::primitive_attr &matmul_attr,
        int32_t weight_cache_type) {

    auto &hash_values = get_onednn_hash_values();
    auto &matmul_weight_cache = get_onednn_matmul_weight_cache();
    std::lock_guard<std::mutex> lock(get_onednn_blocked_weight_mutex());

    dnnl::memory cached_weight_mem;

    // WC=2: identity key on B (no M). Hit reuses packed memory. Miss packs
    // to a temp buffer, memcpy in-place iff blocked size == plain size (and
    // 64-byte pad) for BF16/F16/S8; else keep OOP. Always LRU-insert.
    if (weight_cache_type == 2) {
        const Key_matmul identity_key
                = make_identity_weight_key(transB, K, N, ldb, dnnl_params);
        if (matmul_weight_cache.try_get(identity_key, cached_weight_mem)) {
            dnnl_params.weights.mem = cached_weight_mem;
            apilog_info("Read onednn cached weights (cache hit)");
            return;
        }

        dnnl::memory::desc dnnl_weight_desc
                = onednn_utils_t::to_dnnl_tensor(dnnl_params.weights, eng);
        dnnl::matmul::primitive_desc matmul_pd
                = create_blocked_matmul_pd(dnnl_params, eng, matmul_attr);
        dnnl::memory::desc dnnl_blocked_weight_desc = matmul_pd.weights_desc();

        const size_t blocked_size = dnnl_blocked_weight_desc.get_size();
        const size_t plain_size = dnnl_weight_desc.get_size();
        constexpr size_t alignment = 64;
        const size_t reorder_size
                = (blocked_size + alignment - 1) & ~(alignment - 1);
        const bool inplace_dtype
                = dnnl_params.weights.dtype == data_type_t::bf16
                || dnnl_params.weights.dtype == data_type_t::f16
                || dnnl_params.weights.dtype == data_type_t::s8;
        // KNOWN ISSUE (INT8 WC=2 inplace): s8 blocked size is larger than
        // the user K*N tensor (pad / compensation), so in_place is false
        // and INT8/DA8W8 stay out-of-place. To be fixed later (declared
        // capacity or a layout that fits in B).
        const bool in_place = blocked_size == plain_size
                && reorder_size == plain_size && inplace_dtype;

        dnnl::memory dnnl_weight_mem = dnnl::memory(
                dnnl_weight_desc, eng, dnnl_params.weights.buffer);
        dnnl::memory dnnl_blocked_weight_mem
                = dnnl::memory(dnnl_blocked_weight_desc, eng);

        dnnl::stream eng_stream(eng);
        reorder(dnnl_weight_mem, dnnl_blocked_weight_mem)
                .execute(eng_stream, dnnl_weight_mem, dnnl_blocked_weight_mem);
        eng_stream.wait();

        if (in_place) {
            std::memcpy(dnnl_params.weights.buffer,
                    dnnl_blocked_weight_mem.get_data_handle(), blocked_size);
            dnnl_params.weights.mem = dnnl::memory(
                    dnnl_blocked_weight_desc, eng, dnnl_params.weights.buffer);
        } else {
            if (!inplace_dtype) {
                apilog_info(
                        "onednn WEIGHT_CACHE_IN_PLACE supports BF16, F16, and "
                        "INT8 weights; falling back to out-of-place");
            }
            dnnl_params.weights.mem = dnnl_blocked_weight_mem;
        }

        apilog_info(in_place ? "onednn reorder weights (WEIGHT_CACHE_IN_PLACE, "
                               "adding to cache)"
                             : "onednn reorder weights (adding to cache)");
        matmul_weight_cache.add(identity_key, dnnl_params.weights.mem);
        return;
    }

    // WC=1 (and WC=0 pack): out-of-place reorder. Two-level lookup (full key
    // including M, then blocking hash). WC=0 returns after reorder without add.
    const Key_matmul full_key(transA, transB, M, K, N, lda, ldb,
            dnnl_params.weights.buffer,
            static_cast<uint32_t>(matmul_algo_t::onednn_blocked));

    auto hash_it = hash_values.find(full_key);
    if (hash_it != hash_values.end()) {
        const Key_matmul cache_key(transB, K, N, ldb,
                dnnl_params.weights.buffer,
                static_cast<uint32_t>(matmul_algo_t::onednn_blocked),
                hash_it->second);
        if (matmul_weight_cache.try_get(cache_key, dnnl_params.weights.mem)) {
            apilog_info("Read onednn cached weights (cache hit)");
            return;
        }
        hash_values.erase(hash_it);
    }

    dnnl::memory::desc dnnl_weight_desc
            = onednn_utils_t::to_dnnl_tensor(dnnl_params.weights, eng);
    dnnl::matmul::primitive_desc matmul_pd
            = create_blocked_matmul_pd(dnnl_params, eng, matmul_attr);
    const size_t blocking_hash = hashBlockingDesc(matmul_pd.weights_desc());
    const Key_matmul cache_key(transB, K, N, ldb, dnnl_params.weights.buffer,
            static_cast<uint32_t>(matmul_algo_t::onednn_blocked),
            blocking_hash);

    if (matmul_weight_cache.try_get(cache_key, dnnl_params.weights.mem)) {
        hash_values[full_key] = blocking_hash;
        apilog_info("Read onednn cached weights (blocking hash match)");
        return;
    }

    dnnl::memory dnnl_weight_mem
            = dnnl::memory(dnnl_weight_desc, eng, dnnl_params.weights.buffer);
    dnnl::memory dnnl_blocked_weight_mem
            = dnnl::memory(matmul_pd.weights_desc(), eng);

    dnnl::stream eng_stream(eng);
    reorder(dnnl_weight_mem, dnnl_blocked_weight_mem)
            .execute(eng_stream, dnnl_weight_mem, dnnl_blocked_weight_mem);
    eng_stream.wait();

    dnnl_params.weights.mem = dnnl_blocked_weight_mem;
    if (weight_cache_type == 0) {
        apilog_info("onednn reorder weights (WEIGHT_CACHE_DISABLE)");
        return;
    }
    apilog_info("onednn reorder weights (adding to cache)");
    hash_values[full_key] = blocking_hash;
    matmul_weight_cache.add(cache_key, dnnl_params.weights.mem);
}

void matmul_onednn_wrapper(char transA, char transB, int M, int N, int K,
        float alpha, const void *A, int lda, const void *B, int ldb, float beta,
        void *C, int ldc, matmul_params &lowoha_params,
        matmul_batch_params_t &batch_params, const void *bias,
        zendnnl::common::matmul_algo_t &kernel, size_t src_batch_stride,
        size_t weight_batch_stride, size_t dst_batch_stride) {
    int32_t weight_cache_type
            = effective_weight_cache_type(lowoha_params.weight_cache_type);
    onednn_utils_t::onednn_matmul_params dnnl_params;

    dnnl_params.src.buffer = const_cast<void *>(A);
    dnnl_params.weights.buffer = const_cast<void *>(B);
    dnnl_params.dst.buffer = C;

    dnnl_params.src.dtype = lowoha_params.dtypes.src;
    dnnl_params.weights.dtype = lowoha_params.dtypes.wei;
    dnnl_params.dst.dtype = lowoha_params.dtypes.dst;

    if (bias != nullptr) {
        dnnl_params.bias.buffer = const_cast<void *>(bias);
        dnnl_params.bias.dtype = lowoha_params.dtypes.bias;
    }

    int batch_count = std::max(batch_params.Batch_A, batch_params.Batch_B);
    if (batch_count == 1) {
        dnnl_params.src.dims = {M, K};
        dnnl_params.weights.dims = {K, N};
        dnnl_params.dst.dims = {M, N};
        if (bias != nullptr) dnnl_params.bias.dims = {1, N};
    } else {
        dnnl_params.src.dims = {batch_params.Batch_A, M, K};
        dnnl_params.weights.dims = {batch_params.Batch_B, K, N};
        dnnl_params.dst.dims = {batch_count, M, N};
        if (bias != nullptr) dnnl_params.bias.dims = {1, 1, N};
    }

    dnnl_params.src.is_transposed = (transA == 'n') ? false : true;
    dnnl_params.weights.is_transposed = (transB == 'n') ? false : true;

    if (batch_count == 1) {
        dnnl_params.src.format_tag = (transA == 'n') ? "ab" : "ba";
        dnnl_params.src.strides = (transA == 'n')
                ? std::vector<int64_t> {lda, 1}
                : std::vector<int64_t> {1, lda};
        dnnl_params.weights.format_tag = (transB == 'n') ? "ab" : "ba";
        dnnl_params.weights.strides = (transB == 'n')
                ? std::vector<int64_t> {ldb, 1}
                : std::vector<int64_t> {1, ldb};
        dnnl_params.dst.format_tag = "ab";
        dnnl_params.dst.strides = std::vector<int64_t> {ldc, 1};
        if (bias != nullptr) {
            dnnl_params.bias.format_tag = "ab";
            dnnl_params.bias.strides = std::vector<int64_t> {0, 1};
        }
    } else {
        // oneDNN dims (dnnl::memory::dims) are std::vector<int64_t>; use int64_t
        // explicitly (`long` is 32-bit on MSVC, which would mismatch the dims type).
        int64_t src_stride = static_cast<int64_t>(src_batch_stride);
        int64_t wei_stride = static_cast<int64_t>(weight_batch_stride);
        int64_t dst_stride = static_cast<int64_t>(dst_batch_stride);

        dnnl_params.src.format_tag = (transA == 'n') ? "abc" : "acb";
        dnnl_params.src.strides = (transA == 'n')
                ? std::vector<int64_t> {src_stride, lda, 1}
                : std::vector<int64_t> {src_stride, 1, lda};

        dnnl_params.weights.format_tag = (transB == 'n') ? "abc" : "acb";
        dnnl_params.weights.strides = (transB == 'n')
                ? std::vector<int64_t> {wei_stride, ldb, 1}
                : std::vector<int64_t> {wei_stride, 1, ldb};

        dnnl_params.dst.format_tag = "abc";
        dnnl_params.dst.strides = std::vector<int64_t> {dst_stride, ldc, 1};

        if (bias != nullptr) {
            dnnl_params.bias.format_tag = "abc";
            dnnl_params.bias.strides = std::vector<int64_t> {0, 0, 1};
        }
    }

    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    std::unordered_map<int, dnnl::memory> matmul_args;
    dnnl::primitive_attr matmul_attr;
    dnnl::post_ops matmul_pops;
    int post_op_index = 0;

    if (alpha != 1.0f) {
        matmul_attr.set_scales_mask(DNNL_ARG_SRC, 0);
        auto alpha_mem = dnnl::memory({{1}, dnnl::memory::data_type::f32, {1}},
                eng, const_cast<float *>(&alpha));
        matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, alpha_mem});
    }

    if (beta != 0.0f) {
        matmul_pops.append_sum(beta);
        post_op_index++;
    }

    if (lowoha_params.quant_params.src_scale.buff) {
        dnnl_params.src_quant.scales
                = lowoha_params.quant_params.src_scale.buff;
        dnnl_params.src_quant.scale_dtype
                = lowoha_params.quant_params.src_scale.dt;
        dnnl_params.src_quant.scale_size
                = lowoha_params.quant_params.src_scale.dims;
        matmul_attr.set_scales_mask(DNNL_ARG_SRC,
                dnnl_params.src_quant.scale_size.back() == 1 ? 0 : 1 << 1);

        if (lowoha_params.quant_params.src_zp.buff) {
            dnnl_params.src_quant.zero_points
                    = lowoha_params.quant_params.src_zp.buff;
            dnnl_params.src_quant.zero_dtype
                    = lowoha_params.quant_params.src_zp.dt;
            dnnl_params.src_quant.zero_size
                    = lowoha_params.quant_params.src_zp.dims;
            matmul_attr.set_zero_points_mask(DNNL_ARG_SRC,
                    dnnl_params.src_quant.zero_size.back() == 1 ? 0 : 1 << 1);
        }
    }

    if (lowoha_params.quant_params.wei_scale.buff) {
        dnnl_params.weights_quant.scales
                = lowoha_params.quant_params.wei_scale.buff;
        dnnl_params.weights_quant.scale_dtype
                = lowoha_params.quant_params.wei_scale.dt;
        dnnl_params.weights_quant.scale_size
                = lowoha_params.quant_params.wei_scale.dims;
        matmul_attr.set_scales_mask(DNNL_ARG_WEIGHTS,
                dnnl_params.weights_quant.scale_size.back() == 1 ? 0 : 1 << 1);

        if (lowoha_params.quant_params.wei_zp.buff) {
            dnnl_params.weights_quant.zero_points
                    = lowoha_params.quant_params.wei_zp.buff;
            dnnl_params.weights_quant.zero_dtype
                    = lowoha_params.quant_params.wei_zp.dt;
            dnnl_params.weights_quant.zero_size
                    = lowoha_params.quant_params.wei_zp.dims;
            matmul_attr.set_zero_points_mask(DNNL_ARG_WEIGHTS,
                    dnnl_params.weights_quant.zero_size.back() == 1 ? 0
                                                                    : 1 << 1);
        }
    }

    if (lowoha_params.quant_params.dst_scale.buff) {
        dnnl_params.dst_quant.scales
                = lowoha_params.quant_params.dst_scale.buff;
        dnnl_params.dst_quant.scale_dtype
                = lowoha_params.quant_params.dst_scale.dt;
        dnnl_params.dst_quant.scale_size
                = lowoha_params.quant_params.dst_scale.dims;
        matmul_attr.set_scales_mask(DNNL_ARG_DST,
                dnnl_params.dst_quant.scale_size.back() == 1 ? 0 : 1 << 1);

        if (lowoha_params.quant_params.dst_zp.buff) {
            dnnl_params.dst_quant.zero_points
                    = lowoha_params.quant_params.dst_zp.buff;
            dnnl_params.dst_quant.zero_dtype
                    = lowoha_params.quant_params.dst_zp.dt;
            dnnl_params.dst_quant.zero_size
                    = lowoha_params.quant_params.dst_zp.dims;
            matmul_attr.set_zero_points_mask(DNNL_ARG_DST,
                    dnnl_params.dst_quant.zero_size.back() == 1 ? 0 : 1 << 1);
        }
    }

    if (lowoha_params.postop_.size() > 0) {
        for (size_t po = 0; po < lowoha_params.postop_.size(); po++) {
            switch (lowoha_params.postop_[po].po_type) {
                case post_op_type_t::elu: {
                    log_info("Adding ELU post-op");
                    matmul_pops.append_eltwise(dnnl::algorithm::eltwise_elu,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::relu: {
                    log_info("Adding ReLU post-op");
                    matmul_pops.append_eltwise(dnnl::algorithm::eltwise_relu,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::leaky_relu: {
                    log_info("Adding Leaky ReLU post-op");
                    matmul_pops.append_eltwise(dnnl::algorithm::eltwise_relu,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::gelu_tanh: {
                    log_info("Adding GELU-Tanh post-op");
                    lowoha_params.postop_[po].alpha = 1.0f;
                    matmul_pops.append_eltwise(
                            dnnl::algorithm::eltwise_gelu_tanh,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::gelu_erf: {
                    log_info("Adding GELU-Erf post-op");
                    lowoha_params.postop_[po].alpha = 1.0f;
                    matmul_pops.append_eltwise(
                            dnnl::algorithm::eltwise_gelu_erf,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::tanh: {
                    log_info("Adding Tanh post-op");
                    lowoha_params.postop_[po].alpha = 1.0f;
                    matmul_pops.append_eltwise(dnnl::algorithm::eltwise_tanh,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::square: {
                    log_info("Adding Square post-op");
                    lowoha_params.postop_[po].alpha = 1.0f;
                    matmul_pops.append_eltwise(dnnl::algorithm::eltwise_square,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::abs: {
                    log_info("Adding Abs post-op");
                    lowoha_params.postop_[po].alpha = 1.0f;
                    matmul_pops.append_eltwise(dnnl::algorithm::eltwise_abs,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::sqrt: {
                    log_info("Adding Sqrt post-op");
                    lowoha_params.postop_[po].alpha = 1.0f;
                    matmul_pops.append_eltwise(dnnl::algorithm::eltwise_sqrt,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::exp: {
                    log_info("Adding Exp post-op");
                    lowoha_params.postop_[po].alpha = 1.0f;
                    matmul_pops.append_eltwise(dnnl::algorithm::eltwise_exp,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::log: {
                    log_info("Adding Log post-op");
                    lowoha_params.postop_[po].alpha = 1.0f;
                    matmul_pops.append_eltwise(dnnl::algorithm::eltwise_log,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::sigmoid: {
                    log_info("Adding Sigmoid post-op");
                    lowoha_params.postop_[po].alpha = 1.0f;
                    matmul_pops.append_eltwise(
                            dnnl::algorithm::eltwise_logistic,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::swish: {
                    log_info("Adding Swish post-op");
                    matmul_pops.append_eltwise(dnnl::algorithm::eltwise_swish,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::mish: {
                    log_info("Adding mish post-op");
                    lowoha_params.postop_[po].alpha = 1.0f;
                    matmul_pops.append_eltwise(dnnl::algorithm::eltwise_mish,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::clip: {
                    log_info("Adding Clip post-op");
                    matmul_pops.append_eltwise(dnnl::algorithm::eltwise_clip,
                            lowoha_params.postop_[po].alpha,
                            lowoha_params.postop_[po].beta);
                    break;
                }
                case post_op_type_t::binary_add: {
                    log_info("Adding Binary Add post-op");
                    std::vector<int64_t> binary_dims;
                    if (lowoha_params.postop_[po].dims.size() == 2
                            && batch_count > 1) {
                        binary_dims = {1, lowoha_params.postop_[po].dims[0],
                                lowoha_params.postop_[po].dims[1]};
                    } else {
                        binary_dims = lowoha_params.postop_[po].dims;
                    }
                    onednn_utils_t::onednn_tensor_params binary_tensor;
                    binary_tensor.dims = binary_dims;
                    binary_tensor.buffer = lowoha_params.postop_[po].buff;
                    binary_tensor.dtype = lowoha_params.postop_[po].dtype;
                    binary_tensor.format_tag
                            = binary_dims.size() == 3 ? "abc" : "ab";

                    auto dnnl_buff_desc = onednn_utils_t::to_dnnl_tensor(
                            binary_tensor, eng);
                    auto dnnl_buff_mem = dnnl::memory(
                            dnnl_buff_desc, eng, binary_tensor.buffer);
                    matmul_pops.append_binary(
                            dnnl::algorithm::binary_add, dnnl_buff_desc);
                    matmul_args.insert(
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
                                            | DNNL_ARG_SRC_1,
                                    dnnl_buff_mem});
                    break;
                }
                case post_op_type_t::binary_mul: {
                    log_info("Adding Binary Mul post-op");
                    std::vector<int64_t> binary_dims;
                    if (lowoha_params.postop_[po].dims.size() == 2
                            && batch_count > 1) {
                        binary_dims = {1, lowoha_params.postop_[po].dims[0],
                                lowoha_params.postop_[po].dims[1]};
                    } else {
                        binary_dims = lowoha_params.postop_[po].dims;
                    }
                    onednn_utils_t::onednn_tensor_params binary_tensor;
                    binary_tensor.dims = binary_dims;
                    binary_tensor.buffer = lowoha_params.postop_[po].buff;
                    binary_tensor.dtype = lowoha_params.postop_[po].dtype;
                    binary_tensor.format_tag
                            = binary_dims.size() == 3 ? "abc" : "ab";

                    auto dnnl_buff_desc = onednn_utils_t::to_dnnl_tensor(
                            binary_tensor, eng);
                    auto dnnl_buff_mem = dnnl::memory(
                            dnnl_buff_desc, eng, binary_tensor.buffer);
                    matmul_pops.append_binary(
                            dnnl::algorithm::binary_mul, dnnl_buff_desc);
                    matmul_args.insert(
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
                                            | DNNL_ARG_SRC_1,
                                    dnnl_buff_mem});
                    break;
                }
                default: break;
            }
            post_op_index++;
        }
    }

    if (matmul_pops.len() > 0) { matmul_attr.set_post_ops(matmul_pops); }

    if (kernel == matmul_algo_t::onednn_blocked) {
        getOrCreateBlockedWeights(transA == 't', transB == 't', M, K, N, lda,
                ldb, dnnl_params, eng, matmul_attr, weight_cache_type);
        dnnl_params.is_blocked = true;
    }

    dnnl_params.algo = kernel;

    onednn_matmul_execute(dnnl_params, matmul_args, matmul_attr, eng);
}

#endif

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
