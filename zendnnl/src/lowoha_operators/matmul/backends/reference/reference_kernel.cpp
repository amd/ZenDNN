/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "reference_kernel.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::error_handling;

namespace {

/* ---------------------------------------------------------------------------
 * ref_parallel_3d: OpenMP-parallel walk over a 3-D (d0, d1, d2) index space.
 * Modelled on oneDNN's parallel_nd/for_nd.
 *
 * GCC/Clang -- including clang-cl and MinGW -- get the natural `collapse(3)`
 * nest. The genuine Microsoft compiler gets the same nest hand-flattened into
 * a single parallel loop instead, because MSVC's libomp faults on `collapse`
 * over 64-bit trip counts and zendnnl_compat.hpp therefore neutralizes the
 * clause there (see its `#define collapse(x)`). A neutralized nest would
 * parallelize over d0 alone, which is a single iteration -- i.e. serial -- for
 * this kernel's dominant shapes: batch_size == 1, or M == 1 for GEMV.
 *
 * `parallelize` maps to the OpenMP `if` clause, so callers can keep a
 * work-size threshold.
 * ------------------------------------------------------------------------- */
template <typename IdxT, typename BodyT>
inline void ref_parallel_3d(
        IdxT d0, IdxT d1, IdxT d2, bool parallelize, const BodyT &body) {
#if defined(_MSC_VER) && !defined(__clang__)
    const long long n1 = static_cast<long long>(d1);
    const long long n2 = static_cast<long long>(d2);
    const long long n12 = n1 * n2;
    const long long total = static_cast<long long>(d0) * n12;
#pragma omp parallel for if (parallelize)
    for (long long it = 0; it < total; ++it) {
        body(static_cast<IdxT>(it / n12), static_cast<IdxT>((it / n2) % n1),
                static_cast<IdxT>(it % n2));
    }
#else
#pragma omp parallel for collapse(3) if (parallelize)
    for (IdxT i0 = 0; i0 < d0; ++i0) {
        for (IdxT i1 = 0; i1 < d1; ++i1) {
            for (IdxT i2 = 0; i2 < d2; ++i2) {
                body(i0, i1, i2);
            }
        }
    }
#endif
}

template <typename IdxT, typename BodyT>
inline void ref_parallel_3d(IdxT d0, IdxT d1, IdxT d2, const BodyT &body) {
    ref_parallel_3d(d0, d1, d2, true, body);
}

constexpr float SQRT_2_OVER_PI = 0.79788458347320556640625f;
constexpr float SQRT_2_OVER_2 = 0.707106769084930419921875f;
constexpr float FITTING_CONST = 0.044715f;

size_t quant_num_elements(
        const matmul_quantization_params_t::matmul_quant_t &quant) {
    if (quant.buff == nullptr) { return 0; }
    return quant.dims.empty()
            ? 1
            : static_cast<size_t>(compute_product(quant.dims));
}

constexpr size_t kInvalidBatchStride = static_cast<size_t>(-1);

unsigned int resolve_batch_stride_elems(size_t batch_stride, size_t fallback) {
    return static_cast<unsigned int>(
            batch_stride != kInvalidBatchStride ? batch_stride : fallback);
}

float elu_fwd(float x, float alpha) {
    return x > 0 ? x : alpha * (expf(x) - 1.0f);
}

float relu_fwd(float x) {
    return x > 0 ? x : 0.0f;
}

float leaky_relu_fwd(float x, float nslope) {
    return x > 0 ? x : nslope * x;
}

float tanh_fwd(float x) {
    return tanhf(x);
}

float gelu_tanh_fwd(float x) {
    float v = tanh_fwd(SQRT_2_OVER_PI * x * (1.0f + FITTING_CONST * x * x));
    return (0.5 * x * (1.0f + v));
}

float gelu_erf_fwd(float x) {
    float v = x * SQRT_2_OVER_2;
    return 0.5f * x * (1.0f + erff(v));
}

float sigmoid_fwd(float x) {
    return (1.0 / (1.0 + expf(-x)));
}

float swish_fwd(float x, float scale) {
    float scaled_x = x * scale;
    return x * (1.0 / (1.0 + expf(-scaled_x)));
}

float square_fwd(float x) {
    return x * x;
}

float abs_fwd(float x) {
    return x > 0 ? x : -x;
}

float sqrt_fwd(float x) {
    return sqrtf(x);
}

float exp_fwd(float x) {
    return expf(x);
}

float log_fwd(float x) {
    return logf(x);
}

float clip_fwd(float x, float lower, float upper) {
    x = x > lower ? x : lower;
    return x > upper ? upper : x;
}

float mish_fwd(float x) {
    const float softplus
            = std::fmax(x, 0.0f) + std::log1p(std::exp(-std::fabs(x)));
    return x * std::tanh(softplus);
}

float binary_add_fwd(float x, float y, float scale) {
    return x + (y * scale);
}

float binary_mul_fwd(float x, float y, float scale) {
    return x * (y * scale);
}

template <typename... Args>
void apply_eltwise_post_op(float (*post_op_func)(float, Args...), size_t size,
        float *output, Args... args) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = post_op_func(output[i], args...);
    }
}

void apply_softmax(int batch_size, int M, int N, float *output) {
    const uint64_t rows
            = static_cast<uint64_t>(batch_size) * static_cast<uint64_t>(M);
    const uint64_t cols = static_cast<uint64_t>(N);

    for (uint64_t row = 0; row < rows; ++row) {
        // Compute exponentials for the current row
        double sumExp = 0.0;
        std::vector<double> expRow(cols);

        for (uint64_t col = 0; col < cols; ++col) {
            uint64_t index = static_cast<uint64_t>(row) * cols + col;
            expRow[col] = expf((output[index]));
            sumExp += expRow[col];
        }

        // Normalize each exponential by the sum to get softmax probabilities
        for (uint64_t col = 0; col < cols; ++col) {
            size_t index = static_cast<size_t>(row) * cols + col;
            output[index] = expRow[col] / sumExp;
        }
    }
}

status_t apply_binary_post_op(
        const matmul_post_op &po, size_t size, float *output) {
    if (po.buff == nullptr) {
        log_error("Binary post-op buffer is null");
        return status_t::failure;
    }

    const size_t buf_size = po.dims.empty()
            ? 1
            : static_cast<size_t>(compute_product(po.dims));
    const float po_scale = 1.0f;

    if (po.po_type == post_op_type_t::binary_add) {
        for (size_t i = 0; i < size; ++i) {
            float temp = read_and_cast<float>(po.buff, po.dtype, i % buf_size);
            output[i] = binary_add_fwd(output[i], temp, po_scale);
        }
    } else if (po.po_type == post_op_type_t::binary_mul) {
        for (size_t i = 0; i < size; ++i) {
            float temp = read_and_cast<float>(po.buff, po.dtype, i % buf_size);
            output[i] = binary_mul_fwd(output[i], temp, po_scale);
        }
    }

    return status_t::success;
}

status_t apply_eltwise_post_op(
        const matmul_post_op &po, int batch_size, int M, int N, float *output) {
    const size_t size = static_cast<size_t>(batch_size) * M * N;

    switch (po.po_type) {
        case post_op_type_t::elu:
            apply_eltwise_post_op(elu_fwd, size, output, po.alpha);
            break;
        case post_op_type_t::relu:
            apply_eltwise_post_op(relu_fwd, size, output);
            break;
        case post_op_type_t::leaky_relu:
            apply_eltwise_post_op(leaky_relu_fwd, size, output, po.alpha);
            break;
        case post_op_type_t::gelu_tanh:
            apply_eltwise_post_op(gelu_tanh_fwd, size, output);
            break;
        case post_op_type_t::gelu_erf:
            apply_eltwise_post_op(gelu_erf_fwd, size, output);
            break;
        case post_op_type_t::swish:
            apply_eltwise_post_op(swish_fwd, size, output, po.alpha);
            break;
        case post_op_type_t::sigmoid:
            apply_eltwise_post_op(sigmoid_fwd, size, output);
            break;
        case post_op_type_t::tanh:
            apply_eltwise_post_op(tanh_fwd, size, output);
            break;
        case post_op_type_t::softmax:
            apply_softmax(batch_size, M, N, output);
            break;
        case post_op_type_t::square:
            apply_eltwise_post_op(square_fwd, size, output);
            break;
        case post_op_type_t::abs:
            apply_eltwise_post_op(abs_fwd, size, output);
            break;
        case post_op_type_t::sqrt:
            apply_eltwise_post_op(sqrt_fwd, size, output);
            break;
        case post_op_type_t::exp:
            apply_eltwise_post_op(exp_fwd, size, output);
            break;
        case post_op_type_t::log:
            apply_eltwise_post_op(log_fwd, size, output);
            break;
        case post_op_type_t::clip:
            apply_eltwise_post_op(clip_fwd, size, output, po.alpha, po.beta);
            break;
        case post_op_type_t::mish:
            apply_eltwise_post_op(mish_fwd, size, output);
            break;
        default:
            log_error("This post-op is not supported in ref kernel");
            return status_t::unimplemented;
    }

    return status_t::success;
}

status_t apply_post_ops(int batch_size, int M, int N,
        const std::vector<matmul_post_op> &postops, float *accum_buff_f32) {
    LOG_DEBUG_INFO("Apply post ops in matmul_ref kernel");
    const size_t size = static_cast<size_t>(batch_size) * M * N;

    for (const auto &po : postops) {
        if (po.po_type == post_op_type_t::none) { continue; }

        status_t status = status_t::success;
        if (po.po_type == post_op_type_t::binary_add
                || po.po_type == post_op_type_t::binary_mul) {
            status = apply_binary_post_op(po, size, accum_buff_f32);
        } else {
            status = apply_eltwise_post_op(
                    po, batch_size, M, N, accum_buff_f32);
        }

        if (status != status_t::success) { return status; }
    }

    return status_t::success;
}

void quantize_dst(const matmul_quantization_params_t &quant_params,
        size_t num_elements, float *accum_buff_f32) {
    if (quant_params.dst_scale.buff == nullptr) { return; }

    const size_t dst_scale_size = quant_num_elements(quant_params.dst_scale);
    const size_t dst_zp_size = quant_num_elements(quant_params.dst_zp);

    for (size_t i = 0; i < num_elements; ++i) {
        accum_buff_f32[i] *= read_and_cast<float>(quant_params.dst_scale.buff,
                quant_params.dst_scale.dt, i % dst_scale_size);
        if (quant_params.dst_zp.buff != nullptr && dst_zp_size != 0) {
            accum_buff_f32[i]
                    += read_and_cast<int32_t>(quant_params.dst_zp.buff,
                            quant_params.dst_zp.dt, i % dst_zp_size);
        }
    }
}

void store_output(int BS, int M, int N, int ldc, unsigned int offset_out,
        float *accum_buff_f32, void *output, data_type_t output_dtype) {
    LOG_DEBUG_INFO("Storing matmul_ref kernel output");

    const size_t accum_batch_stride = static_cast<size_t>(M) * N;
    const size_t output_batch_stride = static_cast<size_t>(offset_out);

    const bool parallelize = BS * M * N > 10000;

    if (output_dtype == data_type_t::u8) {
        auto *out_u8 = static_cast<uint8_t *>(output);
        ref_parallel_3d(BS, M, N, parallelize, [&](int bs, int i, int j) {
            const size_t ac_idx = static_cast<size_t>(bs) * accum_batch_stride
                    + static_cast<size_t>(i) * N + static_cast<size_t>(j);
            const size_t out_idx = static_cast<size_t>(bs) * output_batch_stride
                    + static_cast<size_t>(i) * ldc + static_cast<size_t>(j);
            float val = accum_buff_f32[ac_idx];
            val = (val < 0.0f) ? 0.0f
                               : ((val > static_cast<float>(UINT8_MAX))
                                                 ? static_cast<float>(UINT8_MAX)
                                                 : val);
            out_u8[out_idx] = static_cast<uint8_t>(std::nearbyint(val));
        });
    } else if (output_dtype == data_type_t::s8) {
        auto *out_s8 = static_cast<int8_t *>(output);
        ref_parallel_3d(BS, M, N, parallelize, [&](int bs, int i, int j) {
            const size_t ac_idx = static_cast<size_t>(bs) * accum_batch_stride
                    + static_cast<size_t>(i) * N + static_cast<size_t>(j);
            const size_t out_idx = static_cast<size_t>(bs) * output_batch_stride
                    + static_cast<size_t>(i) * ldc + static_cast<size_t>(j);
            float val = accum_buff_f32[ac_idx];
            val = (val < static_cast<float>(INT8_MIN))
                    ? static_cast<float>(INT8_MIN)
                    : ((val > static_cast<float>(INT8_MAX))
                                      ? static_cast<float>(INT8_MAX)
                                      : val);
            out_s8[out_idx] = static_cast<int8_t>(std::nearbyint(val));
        });
    } else if (output_dtype == data_type_t::s32) {
        auto *out_s32 = static_cast<int32_t *>(output);
        ref_parallel_3d(BS, M, N, parallelize, [&](int bs, int i, int j) {
            const size_t ac_idx = static_cast<size_t>(bs) * accum_batch_stride
                    + static_cast<size_t>(i) * N + static_cast<size_t>(j);
            const size_t out_idx = static_cast<size_t>(bs) * output_batch_stride
                    + static_cast<size_t>(i) * ldc + static_cast<size_t>(j);
            float val = accum_buff_f32[ac_idx];
            val = (val < static_cast<float>(INT32_MIN))
                    ? static_cast<float>(INT32_MIN)
                    : ((val > static_cast<float>(INT32_MAX))
                                      ? static_cast<float>(INT32_MAX)
                                      : val);
            out_s32[out_idx] = static_cast<int32_t>(std::nearbyint(val));
        });
    } else if (output_dtype == data_type_t::bf16) {
        auto *out_bf16 = static_cast<bfloat16_t *>(output);
        ref_parallel_3d(BS, M, N, parallelize, [&](int bs, int i, int j) {
            const size_t ac_idx = static_cast<size_t>(bs) * accum_batch_stride
                    + static_cast<size_t>(i) * N + static_cast<size_t>(j);
            const size_t out_idx = static_cast<size_t>(bs) * output_batch_stride
                    + static_cast<size_t>(i) * ldc + static_cast<size_t>(j);
            out_bf16[out_idx] = bfloat16_t(accum_buff_f32[ac_idx]);
        });
    } else if (output_dtype == data_type_t::f16) {
        auto *out_f16 = static_cast<float16_t *>(output);
        ref_parallel_3d(BS, M, N, parallelize, [&](int bs, int i, int j) {
            const size_t ac_idx = static_cast<size_t>(bs) * accum_batch_stride
                    + static_cast<size_t>(i) * N + static_cast<size_t>(j);
            const size_t out_idx = static_cast<size_t>(bs) * output_batch_stride
                    + static_cast<size_t>(i) * ldc + static_cast<size_t>(j);
            out_f16[out_idx] = float16_t(accum_buff_f32[ac_idx]);
        });
    } else {
        auto *out_f32 = static_cast<float *>(output);
        if (BS == 1 && ldc == N) {
            const size_t total_elements = static_cast<size_t>(M) * N;
#pragma omp parallel for if (total_elements > 10000)
            for (size_t idx = 0; idx < total_elements; ++idx) {
                out_f32[idx] = accum_buff_f32[idx];
            }
        } else {
            ref_parallel_3d(BS, M, N, parallelize, [&](int bs, int i, int j) {
                const size_t ac_idx
                        = static_cast<size_t>(bs) * accum_batch_stride
                        + static_cast<size_t>(i) * N + static_cast<size_t>(j);
                const size_t out_idx
                        = static_cast<size_t>(bs) * output_batch_stride
                        + static_cast<size_t>(i) * ldc + static_cast<size_t>(j);
                out_f32[out_idx] = accum_buff_f32[ac_idx];
            });
        }
    }
}

} // namespace

// Extract nibble from packed 4-bit byte (low nibble if is_low_nibble=true, else high nibble)
// For s4 (signed int4): sign-extends from bit 3, yielding range [-8, 7]
// For u4 (unsigned int4): returns raw nibble, yielding range [0, 15]
inline int8_t extract_4bit_nibble(
        int8_t packed_byte, bool is_low_nibble, data_type_t dt) {
    uint8_t ubyte = static_cast<uint8_t>(packed_byte);
    int8_t value = is_low_nibble ? (ubyte & 0x0F) : ((ubyte >> 4) & 0x0F);
    if (dt == data_type_t::s4 && (value & 0x08)) { value |= 0xF0; }
    return value;
}

void compute_zero_point_compensation(int M, int N, int K, const void *src,
        data_type_t input_dtype, int src_s0, int src_s1, int8_t *wei,
        int wei_s0, int wei_s1, int32_t *&zp_comp, int32_t src_zero_point,
        int32_t wei_zero_point, int &zp_comp_size) {
    LOG_DEBUG_INFO(
            "Calculating zero-point compensation in zero_point_compensation");

    if (!wei_zero_point && !src_zero_point) {
        return;
    } else if (!wei_zero_point) {
        // zp_comp is freed in main function
        size_t alignment = 64;
        size_t comp_size
                = (N * sizeof(int32_t) + alignment - 1) & ~(alignment - 1);
        zp_comp = (int32_t *)zendnnl_aligned_alloc(64, comp_size);
        if (!zp_comp) {
            log_error("Failed to allocate zero-point compensation buffer");
            return;
        }
        std::vector<int32_t> wei_comp(N, 0);
        zp_comp_size = N;

        for (auto k = 0; k < K; ++k) {
            for (auto n = 0; n < N; ++n) {
                if (k == 0) { wei_comp[n] = int32_t(0); }
                wei_comp[n] += wei[wei_s0 * k + wei_s1 * n];
            }
        }
        for (auto n = 0; n < N; ++n) {
            zp_comp[n] = 0 - src_zero_point * wei_comp[n];
        }
    } else if (!src_zero_point) {
        std::vector<int32_t> src_comp(M, 0);
        // zp_comp is freed in main function
        size_t alignment = 64;
        size_t comp_size
                = (static_cast<size_t>(M) * N * sizeof(int32_t) + alignment - 1)
                & ~(alignment - 1);
        zp_comp = (int32_t *)zendnnl_aligned_alloc(64, comp_size);
        if (!zp_comp) {
            log_error("Failed to allocate zero-point compensation buffer");
            return;
        }
        zp_comp_size = M * N;

        for (auto m = 0; m < M; ++m) {
            for (auto k = 0; k < K; ++k) {
                if (k == 0) { src_comp[m] = int32_t(0); }
                src_comp[m] += read_and_cast<int32_t>(
                        src, input_dtype, src_s0 * m + src_s1 * k);
            }
        }

        for (auto m = 0; m < M; ++m) {
            for (auto n = 0; n < N; ++n) {
                zp_comp[m * N + n] = 0 - wei_zero_point * src_comp[m];
            }
        }
    } else {
        std::vector<int32_t> src_comp(M, 0);
        std::vector<int32_t> wei_comp(N, 0);
        // zp_comp is freed in main function
        size_t alignment = 64;
        size_t comp_size
                = (static_cast<size_t>(M) * N * sizeof(int32_t) + alignment - 1)
                & ~(alignment - 1);
        zp_comp = (int32_t *)zendnnl_aligned_alloc(64, comp_size);
        if (!zp_comp) {
            log_error("Failed to allocate zero-point compensation buffer");
            return;
        }
        zp_comp_size = M * N;
        //Src comp
        for (auto m = 0; m < M; ++m) {
            for (auto k = 0; k < K; ++k) {
                if (k == 0) { src_comp[m] = int32_t(0); }
                src_comp[m] += read_and_cast<int32_t>(
                        src, input_dtype, src_s0 * m + src_s1 * k);
            }
        }

        for (auto k = 0; k < K; ++k) {
            for (auto n = 0; n < N; ++n) {
                if (k == 0) { wei_comp[n] = int32_t(0); }
                wei_comp[n] += wei[wei_s0 * k + wei_s1 * n];
            }
        }

        for (auto m = 0; m < M; ++m) {
            for (auto n = 0; n < N; ++n) {
                zp_comp[m * N + n] = 0 - src_zero_point * wei_comp[n]
                        - wei_zero_point * src_comp[m]
                        + src_zero_point * wei_zero_point * (int)K;
            }
        }
    }
}

void compute_quantized_matmul(int batch_size, int M, int N, int K, int lda,
        int ldb, int ldc, unsigned int offset_src, unsigned int offset_wei,
        unsigned int offset_out, int batch_a_count, int batch_b_count,
        float alpha, float beta, bool is_transpose_src,
        bool is_transpose_weights, const void *input, const void *weights,
        const void *bias, void *output, float *accum_buff_f32,
        data_type_t input_dtype, data_type_t weight_dtype,
        data_type_t bias_dtype, data_type_t output_dtype,
        const matmul_quantization_params_t &quant_param) {

    LOG_DEBUG_INFO("Computing quantized MatMul_ref");

    const size_t src_scale_size = quant_num_elements(quant_param.src_scale);
    const size_t wei_scale_size = quant_num_elements(quant_param.wei_scale);

    bool src_scale_per_token
            = (src_scale_size == static_cast<size_t>(M) && M > 1);
    bool src_scale_per_group = (src_scale_size > static_cast<size_t>(M)
            && src_scale_size % static_cast<size_t>(M) == 0);
    int src_num_groups
            = src_scale_per_group ? static_cast<int>(src_scale_size) / M : 1;
    int src_group_size = src_scale_per_group ? K / src_num_groups : K;
    int wei_num_groups = 1;
    bool wei_scale_per_group = (wei_scale_size > static_cast<size_t>(N)
            && wei_scale_size % static_cast<size_t>(N) == 0);
    if (wei_scale_per_group) {
        wei_num_groups
                = static_cast<int>(wei_scale_size / static_cast<size_t>(N));
        if (src_scale_per_group && wei_num_groups != src_num_groups) {
            wei_scale_per_group = false;
            wei_num_groups = 1;
        }
    }

    int32_t *zp_comp = nullptr;
    int zp_comp_size = 0;
    int32_t src_zero_point = 0;
    if (quant_param.src_zp.buff || quant_param.wei_zp.buff) {
        src_zero_point = quant_param.src_zp.buff != nullptr
                ? read_and_cast<int32_t>(
                          quant_param.src_zp.buff, quant_param.src_zp.dt)
                : 0;
        int32_t wei_zero_point = quant_param.wei_zp.buff != nullptr
                ? read_and_cast<int32_t>(
                          quant_param.wei_zp.buff, quant_param.wei_zp.dt)
                : 0;
        int src_0 = is_transpose_src ? 1 : lda;
        int src_1 = is_transpose_src ? lda : 1;
        int wei_0 = is_transpose_weights ? 1 : ldb;
        int wei_1 = is_transpose_weights ? ldb : 1;

        compute_zero_point_compensation(M, N, K, input, input_dtype, src_0,
                src_1, (int8_t *)weights, wei_0, wei_1, zp_comp, src_zero_point,
                wei_zero_point, zp_comp_size);
    }

    const size_t accum_batch_stride = static_cast<size_t>(M) * N;

    ref_parallel_3d(batch_size, M, N, [&](int bs, int i, int j) {
        size_t op_idx = static_cast<size_t>(bs) * offset_out
                + static_cast<size_t>(i) * ldc + j;
        size_t ac_idx = static_cast<size_t>(bs) * accum_batch_stride
                + static_cast<size_t>(i) * N + j;
        float sum = 0.0f;

        if (src_scale_per_group) {
            for (int g = 0; g < src_num_groups; ++g) {
                int32_t group_sum = 0;
                int k_start = g * src_group_size;
                int k_end = k_start + src_group_size;
                for (int kk = k_start; kk < k_end; ++kk) {
                    size_t wt_idx = is_transpose_weights
                            ? (static_cast<size_t>(
                                       get_batch_index(bs, batch_b_count))
                                              * offset_wei
                                      + static_cast<size_t>(j) * ldb + kk)
                            : (static_cast<size_t>(
                                       get_batch_index(bs, batch_b_count))
                                              * offset_wei
                                      + static_cast<size_t>(kk) * ldb + j);
                    size_t ip_idx = is_transpose_src
                            ? (static_cast<size_t>(
                                       get_batch_index(bs, batch_a_count))
                                              * offset_src
                                      + static_cast<size_t>(kk) * lda + i)
                            : (static_cast<size_t>(
                                       get_batch_index(bs, batch_a_count))
                                              * offset_src
                                      + static_cast<size_t>(i) * lda + kk);
                    int32_t src_val;
                    if (input_dtype == data_type_t::bf16
                            || input_dtype == data_type_t::f32) {
                        float ip_f32 = read_and_cast<float>(
                                input, input_dtype, ip_idx);
                        float grp_src_scale = read_and_cast<float>(
                                quant_param.src_scale.buff,
                                quant_param.src_scale.dt,
                                i * src_num_groups + g);
                        if (grp_src_scale == 0.f) { grp_src_scale = 1.f; }
                        src_val = static_cast<int32_t>(std::nearbyint(
                                          ip_f32 / grp_src_scale))
                                + src_zero_point;
                    } else {
                        src_val = read_and_cast<int32_t>(
                                input, input_dtype, ip_idx);
                    }
                    group_sum += src_val
                            * read_and_cast<int32_t>(
                                    weights, weight_dtype, wt_idx);
                }
                float grp_scale = read_and_cast<float>(
                        quant_param.src_scale.buff, quant_param.src_scale.dt,
                        i * src_num_groups + g);
                float grp_val = static_cast<float>(group_sum) * grp_scale;
                if (wei_scale_per_group) {
                    grp_val *= read_and_cast<float>(quant_param.wei_scale.buff,
                            quant_param.wei_scale.dt, g * N + j);
                }
                sum += grp_val;
            }
        } else {
            int32_t sum_s32 = 0;
            for (auto k = 0; k < K; ++k) {
                size_t wt_idx = is_transpose_weights
                        ? (static_cast<size_t>(
                                   get_batch_index(bs, batch_b_count))
                                          * offset_wei
                                  + static_cast<size_t>(j) * ldb + k)
                        : (static_cast<size_t>(
                                   get_batch_index(bs, batch_b_count))
                                          * offset_wei
                                  + static_cast<size_t>(k) * ldb + j);
                size_t ip_idx = is_transpose_src
                        ? (static_cast<size_t>(
                                   get_batch_index(bs, batch_a_count))
                                          * offset_src
                                  + static_cast<size_t>(k) * lda + i)
                        : (static_cast<size_t>(
                                   get_batch_index(bs, batch_a_count))
                                          * offset_src
                                  + static_cast<size_t>(i) * lda + k);
                if (input_dtype == data_type_t::bf16
                        || input_dtype == data_type_t::f32) {
                    float ip_f32
                            = read_and_cast<float>(input, input_dtype, ip_idx);
                    size_t src_scl_idx
                            = src_scale_per_token ? static_cast<size_t>(i) : 0;
                    float src_scale
                            = read_and_cast<float>(quant_param.src_scale.buff,
                                    quant_param.src_scale.dt, src_scl_idx);
                    if (src_scale == 0.f) { src_scale = 1.f; }
                    int32_t ip_s32 = static_cast<int32_t>(
                                             std::nearbyint(ip_f32 / src_scale))
                            + src_zero_point;
                    sum_s32 += ip_s32
                            * read_and_cast<int32_t>(
                                    weights, weight_dtype, wt_idx);
                } else {
                    sum_s32 += read_and_cast<int32_t>(
                                       input, input_dtype, ip_idx)
                            * read_and_cast<int32_t>(
                                    weights, weight_dtype, wt_idx);
                }
            }
            sum = static_cast<float>(sum_s32);
        }

        if (alpha != 1.0f) { sum *= alpha; }
        if (beta) {
            sum += read_and_cast<float>(output, output_dtype, op_idx) * beta;
        }
        if (zp_comp) { sum += (float)(zp_comp[(i * N + j) % zp_comp_size]); }
        if (src_scale_size && !src_scale_per_group) {
            size_t scale_idx = src_scale_per_token ? static_cast<size_t>(i) : 0;
            sum *= read_and_cast<float>(quant_param.src_scale.buff,
                    quant_param.src_scale.dt, scale_idx);
        }
        if (wei_scale_size && !(src_scale_per_group && wei_scale_per_group)) {
            sum *= read_and_cast<float>(quant_param.wei_scale.buff,
                    quant_param.wei_scale.dt, j % wei_scale_size);
        }
        if (bias) { sum += read_and_cast<float>(bias, bias_dtype, j); }
        accum_buff_f32[ac_idx] = sum;
    });

    if (zp_comp) { zendnnl_aligned_free(zp_comp); }
}

// W4A8: widen packed s4 to K×N s8 (sign-extended nibble codes, no dequant).
#if !ZENDNNL_DEPENDS_AOCLDLP
void cvt_s4_to_s8(const int8_t *weights, int8_t *wei_s8, int k, int n, int ldb,
        bool is_transposed) {
#pragma omp parallel for collapse(2)
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            size_t physical_idx = is_transposed
                    ? (static_cast<size_t>(col) * ldb + row)
                    : (static_cast<size_t>(row) * ldb + col);
            size_t packed_byte_idx = physical_idx / 2;
            bool is_low_nibble = (physical_idx % 2) == 0;

            int8_t s8_value = extract_4bit_nibble(
                    weights[packed_byte_idx], is_low_nibble, data_type_t::s4);
            wei_s8[static_cast<size_t>(row) * n + col] = s8_value;
        }
    }
}
#endif // !ZENDNNL_DEPENDS_AOCLDLP

// Normalize compact per-tensor/per-token src scales to {M,G} for sym-quant.
status_t broadcast_w4a8_src_scale_ref(
        matmul_quantization_params_t &quant_params, int M,
        std::vector<uint8_t> &expanded_src_scale) {
    const auto &src_dims = quant_params.src_scale.dims;
    if (src_dims.size() != 2) {
        log_error(
                "W4A8 reference: source scale dims must be {1,1}, {M,1}, or "
                "{M,G}");
        return status_t::failure;
    }

    if (quant_params.wei_scale.dims.size() != 2) {
        log_error("W4A8 reference: weight scale dims must be {G,N}");
        return status_t::failure;
    }

    const int64_t src_rows = src_dims[0];
    const int64_t src_cols = src_dims[1];
    const int64_t G_dim = quant_params.wei_scale.dims[0];

    const void *scale_buff = quant_params.src_scale.buff;
    if (!scale_buff) {
        log_error("W4A8 reference: source scale buffer is null");
        return status_t::failure;
    }

    if (src_cols != 1) {
        if (src_rows != M || src_cols != G_dim) {
            log_error(
                    "W4A8 reference: per-group source scale dims must be "
                    "{M,G}");
            return status_t::failure;
        }
        return status_t::success;
    }

    const bool is_per_tensor = src_rows == 1;
    if (!is_per_tensor && src_rows != M) {
        log_error("W4A8 reference: source scale rows must be 1 or M");
        return status_t::failure;
    }

    const int64_t target_rows = M;
    const int64_t target_cols = G_dim;
    const size_t elem_size = size_of(quant_params.src_scale.dt);
    expanded_src_scale.resize(
            static_cast<size_t>(target_rows * target_cols) * elem_size);
    const uint8_t *src_scale_src = static_cast<const uint8_t *>(scale_buff);
    uint8_t *expanded = expanded_src_scale.data();

#pragma omp parallel for collapse(2)
    for (int64_t m = 0; m < target_rows; ++m) {
        for (int64_t g = 0; g < target_cols; ++g) {
            const int64_t src_m = is_per_tensor ? 0 : m;
            std::memcpy(expanded
                            + static_cast<size_t>(m * target_cols + g)
                                    * elem_size,
                    src_scale_src + static_cast<size_t>(src_m) * elem_size,
                    elem_size);
        }
    }

    quant_params.src_scale.buff = expanded;
    quant_params.src_scale.dims = {target_rows, target_cols};
    return status_t::success;
}

status_t compute_w4a8_matmul(int batch_size, int M, int N, int K, int lda,
        int ldb, int ldc, unsigned int offset_src, unsigned int offset_wei,
        unsigned int offset_out, int batch_a_count, int batch_b_count,
        float alpha, float beta, bool is_transpose_src,
        bool is_transpose_weights, const void *input, const void *weights,
        const void *bias, void *output, float *accum_buff_f32,
        data_type_t input_dtype, data_type_t bias_dtype,
        data_type_t output_dtype,
        const matmul_quantization_params_t &quant_params) {
    LOG_DEBUG_INFO("Computing W4A8 MatMul_ref");

    matmul_quantization_params_t w4a8_quant_params = quant_params;
    std::vector<uint8_t> expanded_src_scale;
    status_t scale_status = broadcast_w4a8_src_scale_ref(
            w4a8_quant_params, M, expanded_src_scale);
    if (scale_status != status_t::success) { return scale_status; }

    const size_t weight_nelem = static_cast<size_t>(batch_size) * K * N;
    int8_t *wei_s8
            = static_cast<int8_t *>(malloc(weight_nelem * sizeof(int8_t)));
    if (wei_s8 == nullptr) {
        log_error("Failed to allocate W4A8 s8 weight buffer");
        return status_t::unimplemented;
    }

    const int8_t *packed_weights = static_cast<const int8_t *>(weights);
    cvt_s4_to_s8(packed_weights, wei_s8, K, N, ldb, is_transpose_weights);

    const unsigned int offset_wei_s8
            = static_cast<unsigned int>(K) * static_cast<unsigned int>(N);
    compute_quantized_matmul(batch_size, M, N, K, lda, N, ldc, offset_src,
            offset_wei_s8, offset_out, batch_a_count, batch_b_count, alpha,
            beta, is_transpose_src, false, input, wei_s8, bias, output,
            accum_buff_f32, input_dtype, data_type_t::s8, bias_dtype,
            output_dtype, w4a8_quant_params);
    free(wei_s8);
    return status_t::success;
}

void compute_matmul(int batch_size, int M, int N, int K, int lda, int ldb,
        int ldc, unsigned int offset_src, unsigned int offset_wei,
        unsigned int offset_out, int batch_a_count, int batch_b_count,
        float alpha, float beta, bool is_transpose_src,
        bool is_transpose_weights, const void *input, const void *weights,
        const void *bias, const void *output, float *accum_buff_f32,
        data_type_t input_dtype, data_type_t weight_dtype,
        data_type_t bias_dtype, data_type_t output_dtype) {

    LOG_DEBUG_INFO("Computing MatMul_ref");
    // NOTE: When validating AOCL DLP's F16 GEMM, the reference must match DLP's
    // hardware-level FP16 accumulation behavior (FMA with FP16 rounding per step
    // and KC=2048 blocking). Without this, rounding differences can cause false
    // mismatches. For all other algorithms, standard F32 accumulation is used.
    matmul_config_t &matmul_config = matmul_config_t::instance();
    data_type_t accum_type = matmul_config.get_accum_type();
    const bool use_f16_accum
            = (accum_type == data_type_t::f16 && input_dtype == data_type_t::f16
                    && weight_dtype == data_type_t::f16
                    && (output_dtype == data_type_t::f16
                            || output_dtype == data_type_t::f32));

    const size_t accum_batch_stride = static_cast<size_t>(M) * N;

    ref_parallel_3d(batch_size, M, N, [&](int bs, int i, int j) {
        size_t op_idx = static_cast<size_t>(bs) * offset_out
                + static_cast<size_t>(i) * ldc + j;
        size_t ac_idx = static_cast<size_t>(bs) * accum_batch_stride
                + static_cast<size_t>(i) * N + j;

        // F16 accumulation path for FP16 FMA
        if (use_f16_accum) {
            const int KC = 2048;
            float alpha_f32 = static_cast<float>(float16_t(alpha));
            for (int pc = 0; pc < K; pc += KC) {
                int kc0 = std::min(K - pc, KC);

                // Beta is applied for first KC block
                // beta 1 is applied for subsequent blocks
                float16_t beta0 = (pc == 0) ? float16_t(beta) : float16_t(1.0f);

                float16_t sum_f16 = float16_t(0.0f);
                for (int kk = 0; kk < kc0; ++kk) {
                    int k_idx = pc + kk;
                    size_t wt_idx = is_transpose_weights
                            ? (static_cast<size_t>(
                                       get_batch_index(bs, batch_b_count))
                                              * offset_wei
                                      + static_cast<size_t>(j) * ldb + k_idx)
                            : (static_cast<size_t>(
                                       get_batch_index(bs, batch_b_count))
                                              * offset_wei
                                      + static_cast<size_t>(k_idx) * ldb + j);
                    size_t ip_idx = is_transpose_src
                            ? (static_cast<size_t>(
                                       get_batch_index(bs, batch_a_count))
                                              * offset_src
                                      + static_cast<size_t>(k_idx) * lda + i)
                            : (static_cast<size_t>(
                                       get_batch_index(bs, batch_a_count))
                                              * offset_src
                                      + static_cast<size_t>(i) * lda + k_idx);
                    float a_f32
                            = read_and_cast<float>(input, input_dtype, ip_idx);
                    float b_f32 = read_and_cast<float>(
                            weights, weight_dtype, wt_idx);
                    sum_f16 = float16_t(std::fmaf(
                            a_f32, b_f32, static_cast<float>(sum_f16)));
                }

                float sum_f32 = static_cast<float>(sum_f16);
                float beta_f32 = static_cast<float>(float16_t(beta0));

                if (beta_f32) {
                    float c_f32 = (pc == 0)
                            ? read_and_cast<float>(output, output_dtype, op_idx)
                            : accum_buff_f32[ac_idx];
                    accum_buff_f32[ac_idx] = static_cast<float>(
                            float16_t(c_f32 * beta_f32 + sum_f32 * alpha_f32));
                } else {
                    accum_buff_f32[ac_idx] = static_cast<float>(
                            float16_t(sum_f32 * alpha_f32));
                }
            }
            if (bias) {
                float sum = accum_buff_f32[ac_idx];
                accum_buff_f32[ac_idx] = static_cast<float>(float16_t(
                        sum + read_and_cast<float>(bias, bias_dtype, j)));
            }
        } else {
            // Default F32 accumulation path for all other data types
            float sum = 0.0;
            for (auto k = 0; k < K; ++k) {
                size_t wt_idx = is_transpose_weights
                        ? (static_cast<size_t>(
                                   get_batch_index(bs, batch_b_count))
                                          * offset_wei
                                  + static_cast<size_t>(j) * ldb + k)
                        : (static_cast<size_t>(
                                   get_batch_index(bs, batch_b_count))
                                          * offset_wei
                                  + static_cast<size_t>(k) * ldb + j);
                size_t ip_idx = is_transpose_src
                        ? (static_cast<size_t>(
                                   get_batch_index(bs, batch_a_count))
                                          * offset_src
                                  + static_cast<size_t>(k) * lda + i)
                        : (static_cast<size_t>(
                                   get_batch_index(bs, batch_a_count))
                                          * offset_src
                                  + static_cast<size_t>(i) * lda + k);
                sum += read_and_cast<float>(input, input_dtype, ip_idx)
                        * read_and_cast<float>(weights, weight_dtype, wt_idx);
            }

            if (alpha != 1.0f) { sum *= alpha; }
            if (beta) {
                sum += read_and_cast<float>(output, output_dtype, op_idx)
                        * beta;
            }
            if (bias) { sum += read_and_cast<float>(bias, bias_dtype, j); }
            accum_buff_f32[ac_idx] = sum;
        }
    });
}

status_t reference_matmul_execute(const char layout, const bool transA,
        const bool transB, const int M, const int N, const int K,
        const float alpha, const void *src, const int lda, const void *weight,
        const int ldb, const void *bias, const float beta, void *dst,
        const int ldc, const bool is_weights_const,
        matmul_batch_params_t &batch_params, matmul_params &params) {
    LOG_DEBUG_INFO("Executing matmul_ref kernel");

    if (layout != 'r' && layout != 'R') {
        log_error(
                "Reference matmul kernel only supports row-major layout, "
                "layout='",
                layout, "'");
        return status_t::unimplemented;
    }

    const void *input = src;
    const void *weights = weight;
    void *output = dst;

    auto input_dtype = params.dtypes.src;
    auto weight_dtype = params.dtypes.wei;
    auto output_dtype = params.dtypes.dst;
    bool is_transpose_src = transA;
    bool is_transpose_weights = transB;

    const int batch_size = std::max(batch_params.Batch_A, batch_params.Batch_B);
    const int batch_a_count = batch_params.Batch_A;
    const int batch_b_count = batch_params.Batch_B;
    const bool is_batched = batch_size > 1;
    unsigned int offset_src = 0;
    unsigned int offset_wei = 0;
    unsigned int offset_out = 0;
    if (is_batched) {
        offset_src = resolve_batch_stride_elems(batch_params.batch_stride_src,
                is_transpose_src ? static_cast<size_t>(K) * lda
                                 : static_cast<size_t>(M) * lda);
        offset_wei = resolve_batch_stride_elems(batch_params.batch_stride_wei,
                is_transpose_weights ? static_cast<size_t>(N) * ldb
                                     : static_cast<size_t>(K) * ldb);
        offset_out = resolve_batch_stride_elems(batch_params.batch_stride_dst,
                static_cast<size_t>(M) * static_cast<size_t>(ldc));
    }
    bool is_int8
            = (input_dtype == data_type_t::s8 || input_dtype == data_type_t::u8
                      || input_dtype == data_type_t::bf16
                      || input_dtype == data_type_t::f32)
            && weight_dtype == data_type_t::s8;
    bool is_w4a8 = is_w4a8_config(params);
    bool is_woq = !is_w4a8 && input_dtype == data_type_t::bf16
            && (weight_dtype == data_type_t::s4
                    || weight_dtype == data_type_t::u4);
    // Interim buffer  size
    size_t output_size = static_cast<size_t>(batch_size) * M * N;
    // Interim accumulation buffer with float type
    float *accum_buff_f32 = (float *)malloc(output_size * sizeof(float));
    if (accum_buff_f32 == nullptr) {
        log_error("Failed to allocate accumulation buffer");
        return status_t::unimplemented;
    }

    data_type_t bias_dtype = params.dtypes.bias;

    if (is_int8
            && (input_dtype == data_type_t::bf16
                    || input_dtype == data_type_t::f32)
            && params.quant_params.src_scale.buff == nullptr) {
        log_error(
                "INT8 reference matmul requires src_scale for BF16/F32 inputs "
                "with s8 weights");
        free(accum_buff_f32);
        return status_t::failure;
    }
    if (is_int8) {
        compute_quantized_matmul(batch_size, M, N, K, lda, ldb, ldc, offset_src,
                offset_wei, offset_out, batch_a_count, batch_b_count, alpha,
                beta, is_transpose_src, is_transpose_weights, input, weights,
                bias, output, accum_buff_f32, input_dtype, weight_dtype,
                bias_dtype, output_dtype, params.quant_params);
    } else if (is_w4a8) {
        status_t w4a8_status = compute_w4a8_matmul(batch_size, M, N, K, lda,
                ldb, ldc, offset_src, offset_wei, offset_out, batch_a_count,
                batch_b_count, alpha, beta, is_transpose_src,
                is_transpose_weights, input, weights, bias, output,
                accum_buff_f32, input_dtype, bias_dtype, output_dtype,
                params.quant_params);
        if (w4a8_status != status_t::success) {
            free(accum_buff_f32);
            return w4a8_status;
        }
    } else if (is_woq) {
        // WOQ: Weight-Only Quantization (BF16 input, S4/U4 weights)
        // Convert S4/U4 weights to BF16 and use compute_matmul
        bool is_float_domain
                = params.quant_params.wei_zp.dt == data_type_t::bf16;
        // Allocate tensor for dequantized BF16 weights
        size_t weight_nelem = static_cast<size_t>(batch_size) * K * N;
        bfloat16_t *bf16_weights = static_cast<bfloat16_t *>(
                malloc(weight_nelem * sizeof(bfloat16_t)));
        if (bf16_weights == nullptr) {
            log_error("Failed to allocate BF16 weight buffer");
            free(accum_buff_f32);
            return status_t::unimplemented;
        }

        // Dequantize S4 weights to BF16
        // S4 weights are packed: 2 values per byte
        // The tensor's physical layout depends on its order (ab vs ba)
        const int8_t *packed_weights = (const int8_t *)weights;
        const size_t wei_scale_size = params.quant_params.wei_scale.dims.empty()
                ? 1
                : static_cast<size_t>(
                          compute_product(params.quant_params.wei_scale.dims));
        bool has_zp = (params.quant_params.wei_zp.buff != nullptr
                && weight_dtype == data_type_t::u4);

        // Determine quantization granularity:
        // - Per-tensor:  wei_scale_size == 1
        // - Per-channel: wei_scale_size == N
        // - Per-group:   wei_scale_size == G * N, where G = K / group_size
        int num_groups = 1;
        int group_size = K;
        if (wei_scale_size > 1 && wei_scale_size != static_cast<size_t>(N)) {
            // Per-group quantization
            num_groups = wei_scale_size / N;
            group_size = K / num_groups;
        }

        // Use tensor strides for correct S4 data access
        // For S4, ldb represents stride in elements (not bytes)
        ref_parallel_3d(batch_size, K, N, [&](int bs, int k, int n) {
            const int wei_bs = get_batch_index(bs, batch_b_count);
            // Calculate S4 element index using tensor strides
            // For non-transposed (ab): element[k,n] at k*ldb + n
            // For transposed (ba): element[k,n] at n*ldb + k
            size_t unpacked_idx = is_transpose_weights
                    ? (wei_bs * offset_wei + n * ldb + k)
                    : (wei_bs * offset_wei + k * ldb + n);
            size_t packed_byte_idx = unpacked_idx / 2;
            bool is_low_nibble = (unpacked_idx % 2) == 0;

            // Extract s4/u4 value
            int8_t packed_byte = packed_weights[packed_byte_idx];
            int8_t value_4bit = extract_4bit_nibble(
                    packed_byte, is_low_nibble, weight_dtype);

            // Get scale based on quantization granularity
            // - Per-tensor:  index = 0
            // - Per-channel: index = n
            // - Per-group:   index = group_idx * N + n, where group_idx = k / group_size
            size_t scale_idx;
            if (wei_scale_size == 1) {
                scale_idx = 0; // Per-tensor
            } else if (wei_scale_size == static_cast<size_t>(N)) {
                scale_idx = n; // Per-channel
            } else {
                // Per-group: scale[group_idx, n]
                int group_idx = k / group_size;
                scale_idx = group_idx * N + n;
            }
            float scale
                    = read_and_cast<float>(params.quant_params.wei_scale.buff,
                            params.quant_params.wei_scale.dt, scale_idx);

            // Get zero point if available (same indexing logic)
            float zp = 0.0f;
            if (has_zp) {
                size_t zp_idx;
                const size_t zp_size = params.quant_params.wei_zp.dims.empty()
                        ? 1
                        : static_cast<size_t>(compute_product(
                                  params.quant_params.wei_zp.dims));
                if (zp_size == 1) {
                    zp_idx = 0; // Per-tensor
                } else if (zp_size == static_cast<size_t>(N)) {
                    zp_idx = n; // Per-channel
                } else {
                    // Per-group: zp[group_idx, n]
                    int group_idx = k / group_size;
                    zp_idx = group_idx * N + n;
                }
                zp = read_and_cast<float>(params.quant_params.wei_zp.buff,
                        params.quant_params.wei_zp.dt, zp_idx);
            }

            float dequant_value = 0.0f;
            // If the weights are s4 or the domain is not float, use the symmetric quantization formula
            if (weight_dtype == data_type_t::s4 || !is_float_domain) {
                dequant_value = (static_cast<float>(value_4bit) - zp) * scale;
            }
            // If the weights are u4 and the domain is float, use the asymmetric quantization formula
            else if (weight_dtype == data_type_t::u4) {
                dequant_value
                        = (static_cast<float>(value_4bit) - 8) * scale + zp;
            }

            // Store in standard K x N layout (untranspose if needed)
            const size_t bf16_idx = static_cast<size_t>(wei_bs)
                            * static_cast<size_t>(K) * static_cast<size_t>(N)
                    + static_cast<size_t>(k) * static_cast<size_t>(N)
                    + static_cast<size_t>(n);
            bf16_weights[bf16_idx] = bfloat16_t(dequant_value);
        });

        // Dequantized weights are now in standard K x N layout
        int ldb_bf16 = N;
        unsigned int offset_wei_bf16 = K * N;

        // Call compute_matmul with dequantized BF16 weights (no longer transposed)
        compute_matmul(batch_size, M, N, K, lda, ldb_bf16, ldc, offset_src,
                offset_wei_bf16, offset_out, batch_a_count, batch_b_count,
                alpha, beta, is_transpose_src, false, input, bf16_weights, bias,
                output, accum_buff_f32, input_dtype, data_type_t::bf16,
                bias_dtype, output_dtype);
        free(bf16_weights);
    } else {
        compute_matmul(batch_size, M, N, K, lda, ldb, ldc, offset_src,
                offset_wei, offset_out, batch_a_count, batch_b_count, alpha,
                beta, is_transpose_src, is_transpose_weights, input, weights,
                bias, output, accum_buff_f32, input_dtype, weight_dtype,
                bias_dtype, output_dtype);
    }
    if (apply_post_ops(batch_size, M, N, params.postop_, accum_buff_f32)
            != status_t::success) {
        free(accum_buff_f32);
        return status_t::failure;
    }
    if (is_int8) {
        quantize_dst(params.quant_params, output_size, accum_buff_f32);
    }
    store_output(batch_size, M, N, ldc, offset_out, accum_buff_f32, output,
            output_dtype);
    free(accum_buff_f32);

    return status_t::success;
}
} //namespace matmul
} //namespace lowoha
} //namespace zendnnl
