/*******************************************************************************
* Modifications Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2018-2022 Intel Corporation
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

// DO NOT EDIT, AUTO-GENERATED
// Use this script to update the file: scripts/generate_zendnn_debug.py

// clang-format off

#include <assert.h>

#include "zendnn_debug.h"
#include "zendnn_types.h"

const char *zendnn_status2str(zendnn_status_t v) {
    if (v == zendnn_success) {
        return "success";
    }
    if (v == zendnn_out_of_memory) {
        return "out_of_memory";
    }
    if (v == zendnn_invalid_arguments) {
        return "invalid_arguments";
    }
    if (v == zendnn_unimplemented) {
        return "unimplemented";
    }
    if (v == zendnn_iterator_ends) {
        return "iterator_ends";
    }
    if (v == zendnn_runtime_error) {
        return "runtime_error";
    }
    if (v == zendnn_not_required) {
        return "not_required";
    }
    assert(!"unknown status");
    return "unknown status";
}

const char *zendnn_dt2str(zendnn_data_type_t v) {
    if (v == zendnn_data_type_undef) {
        return "undef";
    }
    if (v == zendnn_f16) {
        return "f16";
    }
    if (v == zendnn_bf16) {
        return "bf16";
    }
    if (v == zendnn_f32) {
        return "f32";
    }
    if (v == zendnn_s32) {
        return "s32";
    }
    if (v == zendnn_s16) {
        return "s16";
    }
    if (v == zendnn_s8) {
        return "s8";
    }
    if (v == zendnn_u8) {
        return "u8";
    }
    if (v == zendnn_s4) {
        return "s4";
    }
    if (v == zendnn_u4) {
        return "u4";
    }
    assert(!"unknown dt");
    return "unknown dt";
}

const char *zendnn_fmt_kind2str(zendnn_format_kind_t v) {
    if (v == zendnn_format_kind_undef) {
        return "undef";
    }
    if (v == zendnn_format_kind_any) {
        return "any";
    }
    if (v == zendnn_blocked) {
        return "blocked";
    }
    if (v == zendnn_format_kind_wino) {
        return "wino";
    }
    if (v == zendnn_format_kind_rnn_packed) {
        return "rnn_packed";
    }
    assert(!"unknown fmt_kind");
    return "unknown fmt_kind";
}

const char *zendnn_prop_kind2str(zendnn_prop_kind_t v) {
    if (v == zendnn_prop_kind_undef) {
        return "undef";
    }
    if (v == zendnn_forward_training || v == zendnn_forward) {
        return "forward_training or forward";
    }
    if (v == zendnn_forward_inference || v == zendnn_forward_scoring) {
        return "forward_inference or forward_scoring";
    }
    if (v == zendnn_backward) {
        return "backward";
    }
    if (v == zendnn_backward_data) {
        return "backward_data";
    }
    if (v == zendnn_backward_weights) {
        return "backward_weights";
    }
    if (v == zendnn_backward_bias) {
        return "backward_bias";
    }
    assert(!"unknown prop_kind");
    return "unknown prop_kind";
}

const char *zendnn_prim_kind2str(zendnn_primitive_kind_t v) {
    if (v == zendnn_undefined_primitive) {
        return "undef";
    }
    if (v == zendnn_reorder) {
        return "reorder";
    }
    if (v == zendnn_shuffle) {
        return "shuffle";
    }
    if (v == zendnn_concat) {
        return "concat";
    }
    if (v == zendnn_sum) {
        return "sum";
    }
    if (v == zendnn_convolution) {
        return "convolution";
    }
    if (v == zendnn_deconvolution) {
        return "deconvolution";
    }
    if (v == zendnn_eltwise) {
        return "eltwise";
    }
    if (v == zendnn_softmax) {
        return "softmax";
    }
    if (v == zendnn_pooling) {
        return "pooling";
    }
    if (v == zendnn_lrn) {
        return "lrn";
    }
    if (v == zendnn_batch_normalization) {
        return "batch_normalization";
    }
    if (v == zendnn_layer_normalization) {
        return "layer_normalization";
    }
    if (v == zendnn_inner_product) {
        return "inner_product";
    }
    if (v == zendnn_rnn) {
        return "rnn";
    }
    if (v == zendnn_gemm) {
        return "gemm";
    }
    if (v == zendnn_binary) {
        return "binary";
    }
    if (v == zendnn_logsoftmax) {
        return "logsoftmax";
    }
    if (v == zendnn_matmul) {
        return "matmul";
    }
    if (v == zendnn_resampling) {
        return "resampling";
    }
    if (v == zendnn_pooling_v2) {
        return "pooling_v2";
    }
    if (v == zendnn_reduction) {
        return "reduction";
    }
    if (v == zendnn_prelu) {
        return "prelu";
    }
    if (v == zendnn_softmax_v2) {
        return "softmax_v2";
    }
    if (v == zendnn_embedding_bag) {
        return "embedding_bag";
    }
    if (v == zendnn_primitive_kind_max) {
        return "primitive_kind_max";
    }
    assert(!"unknown prim_kind");
    return "unknown prim_kind";
}

const char *zendnn_alg_kind2str(zendnn_alg_kind_t v) {
    if (v == zendnn_alg_kind_undef) {
        return "undef";
    }
    if (v == zendnn_convolution_gemm) {
        return "convolution_gemm";
    }
    if (v == zendnn_convolution_gemm_bf16bf16f32of32) {
        return "convolution_gemm_bf16bf16f32of32";
    }
    if (v == zendnn_convolution_gemm_bf16bf16f32obf16) {
        return "convolution_gemm_bf16bf16f32obf16";
    }
    if (v == zendnn_convolution_gemm_u8s8s16os16) {
        return "convolution_gemm_u8s8s16os16";
    }
    if (v == zendnn_convolution_gemm_u8s8s16os8) {
        return "convolution_gemm_u8s8s16os8";
    }
    if (v == zendnn_convolution_gemm_u8s8s16ou8) {
        return "convolution_gemm_u8s8s16ou8";
    }
    if (v == zendnn_convolution_gemm_u8s8s32os32) {
        return "convolution_gemm_u8s8s32os32";
    }
    if (v == zendnn_convolution_gemm_u8s8s32os8) {
        return "convolution_gemm_u8s8s32os8";
    }
    if (v == zendnn_convolution_gemm_s8s8s32os32) {
        return "convolution_gemm_s8s8s32os32";
    }
    if (v == zendnn_convolution_gemm_s8s8s32os8) {
        return "convolution_gemm_s8s8s32os8";
    }
    if (v == zendnn_convolution_gemm_s8s8s16os16) {
        return "convolution_gemm_s8s8s16os16";
    }
    if (v == zendnn_convolution_gemm_s8s8s16os8) {
        return "convolution_gemm_s8s8s16os8";
    }
    if (v == zendnn_convolution_ref) {
        return "convolution_ref";
    }
    if (v == zendnn_convolution_ck) {
        return "convolution_ck";
    }
    if (v == zendnn_convolution_direct) {
        return "convolution_direct";
    }
    if (v == zendnn_convolution_winograd) {
        return "convolution_winograd";
    }
    if (v == zendnn_convolution_auto) {
        return "convolution_auto";
    }
    if (v == zendnn_deconvolution_direct) {
        return "deconvolution_direct";
    }
    if (v == zendnn_deconvolution_winograd) {
        return "deconvolution_winograd";
    }
    if (v == zendnn_eltwise_relu) {
        return "eltwise_relu";
    }
    if (v == zendnn_eltwise_tanh) {
        return "eltwise_tanh";
    }
    if (v == zendnn_eltwise_elu) {
        return "eltwise_elu";
    }
    if (v == zendnn_eltwise_square) {
        return "eltwise_square";
    }
    if (v == zendnn_eltwise_abs) {
        return "eltwise_abs";
    }
    if (v == zendnn_eltwise_sqrt) {
        return "eltwise_sqrt";
    }
    if (v == zendnn_eltwise_linear) {
        return "eltwise_linear";
    }
    if (v == zendnn_eltwise_bounded_relu) {
        return "eltwise_bounded_relu";
    }
    if (v == zendnn_eltwise_soft_relu) {
        return "eltwise_soft_relu";
    }
    if (v == zendnn_eltwise_logistic) {
        return "eltwise_logistic";
    }
    if (v == zendnn_eltwise_exp) {
        return "eltwise_exp";
    }
    if (v == zendnn_eltwise_gelu_tanh || v == zendnn_eltwise_gelu) {
        return "eltwise_gelu_tanh or eltwise_gelu";
    }
    if (v == zendnn_eltwise_swish) {
        return "eltwise_swish";
    }
    if (v == zendnn_eltwise_log) {
        return "eltwise_log";
    }
    if (v == zendnn_eltwise_clip) {
        return "eltwise_clip";
    }
    if (v == zendnn_eltwise_clip_v2) {
        return "eltwise_clip_v2";
    }
    if (v == zendnn_eltwise_pow) {
        return "eltwise_pow";
    }
    if (v == zendnn_eltwise_gelu_erf) {
        return "eltwise_gelu_erf";
    }
    if (v == zendnn_eltwise_round) {
        return "eltwise_round";
    }
    if (v == zendnn_eltwise_logsigmoid) {
        return "eltwise_logsigmoid";
    }
    if (v == zendnn_eltwise_mish) {
        return "eltwise_mish";
    }
    if (v == zendnn_eltwise_hardswish) {
        return "eltwise_hardswish";
    }
    if (v == zendnn_eltwise_relu_use_dst_for_bwd) {
        return "eltwise_relu_use_dst_for_bwd";
    }
    if (v == zendnn_eltwise_tanh_use_dst_for_bwd) {
        return "eltwise_tanh_use_dst_for_bwd";
    }
    if (v == zendnn_eltwise_elu_use_dst_for_bwd) {
        return "eltwise_elu_use_dst_for_bwd";
    }
    if (v == zendnn_eltwise_sqrt_use_dst_for_bwd) {
        return "eltwise_sqrt_use_dst_for_bwd";
    }
    if (v == zendnn_eltwise_logistic_use_dst_for_bwd) {
        return "eltwise_logistic_use_dst_for_bwd";
    }
    if (v == zendnn_eltwise_exp_use_dst_for_bwd) {
        return "eltwise_exp_use_dst_for_bwd";
    }
    if (v == zendnn_eltwise_clip_v2_use_dst_for_bwd) {
        return "eltwise_clip_v2_use_dst_for_bwd";
    }
    if (v == zendnn_pooling_max) {
        return "pooling_max";
    }
    if (v == zendnn_pooling_avg_include_padding) {
        return "pooling_avg_include_padding";
    }
    if (v == zendnn_pooling_avg_exclude_padding || v == zendnn_pooling_avg) {
        return "pooling_avg_exclude_padding or pooling_avg";
    }
    if (v == zendnn_lrn_across_channels) {
        return "lrn_across_channels";
    }
    if (v == zendnn_lrn_within_channel) {
        return "lrn_within_channel";
    }
    if (v == zendnn_vanilla_rnn) {
        return "vanilla_rnn";
    }
    if (v == zendnn_vanilla_lstm) {
        return "vanilla_lstm";
    }
    if (v == zendnn_vanilla_gru) {
        return "vanilla_gru";
    }
    if (v == zendnn_lbr_gru) {
        return "lbr_gru";
    }
    if (v == zendnn_vanilla_augru) {
        return "vanilla_augru";
    }
    if (v == zendnn_lbr_augru) {
        return "lbr_augru";
    }
    if (v == zendnn_binary_add) {
        return "binary_add";
    }
    if (v == zendnn_binary_mul) {
        return "binary_mul";
    }
    if (v == zendnn_binary_max) {
        return "binary_max";
    }
    if (v == zendnn_binary_min) {
        return "binary_min";
    }
    if (v == zendnn_binary_div) {
        return "binary_div";
    }
    if (v == zendnn_binary_sub) {
        return "binary_sub";
    }
    if (v == zendnn_binary_ge) {
        return "binary_ge";
    }
    if (v == zendnn_binary_gt) {
        return "binary_gt";
    }
    if (v == zendnn_binary_le) {
        return "binary_le";
    }
    if (v == zendnn_binary_lt) {
        return "binary_lt";
    }
    if (v == zendnn_binary_eq) {
        return "binary_eq";
    }
    if (v == zendnn_binary_ne) {
        return "binary_ne";
    }
    if (v == zendnn_resampling_nearest) {
        return "resampling_nearest";
    }
    if (v == zendnn_resampling_linear) {
        return "resampling_linear";
    }
    if (v == zendnn_reduction_max) {
        return "reduction_max";
    }
    if (v == zendnn_reduction_min) {
        return "reduction_min";
    }
    if (v == zendnn_reduction_sum) {
        return "reduction_sum";
    }
    if (v == zendnn_reduction_mul) {
        return "reduction_mul";
    }
    if (v == zendnn_reduction_mean) {
        return "reduction_mean";
    }
    if (v == zendnn_reduction_norm_lp_max) {
        return "reduction_norm_lp_max";
    }
    if (v == zendnn_reduction_norm_lp_sum) {
        return "reduction_norm_lp_sum";
    }
    if (v == zendnn_reduction_norm_lp_power_p_max) {
        return "reduction_norm_lp_power_p_max";
    }
    if (v == zendnn_reduction_norm_lp_power_p_sum) {
        return "reduction_norm_lp_power_p_sum";
    }
    if (v == zendnn_softmax_accurate) {
        return "softmax_accurate";
    }
    if (v == zendnn_softmax_log) {
        return "softmax_log";
    }
    if (v == zendnn_embedding_bag_sum) {
        return "embedding_bag_sum";
    }
    if (v == zendnn_embedding_bag_mean) {
        return "embedding_bag_mean";
    }
    if (v == zendnn_embedding_bag_max) {
        return "embedding_bag_max";
    }
    assert(!"unknown alg_kind");
    return "unknown alg_kind";
}

const char *zendnn_rnn_flags2str(zendnn_rnn_flags_t v) {
    if (v == zendnn_rnn_flags_undef) {
        return "undef";
    }
    assert(!"unknown rnn_flags");
    return "unknown rnn_flags";
}

const char *zendnn_rnn_direction2str(zendnn_rnn_direction_t v) {
    if (v == zendnn_unidirectional_left2right) {
        return "unidirectional_left2right";
    }
    if (v == zendnn_unidirectional_right2left) {
        return "unidirectional_right2left";
    }
    if (v == zendnn_bidirectional_concat) {
        return "bidirectional_concat";
    }
    if (v == zendnn_bidirectional_sum) {
        return "bidirectional_sum";
    }
    //if (v == zendnn_unidirectional) return "unidirectional";
    assert(!"unknown rnn_direction");
    return "unknown rnn_direction";
}

const char *zendnn_engine_kind2str(zendnn_engine_kind_t v) {
    if (v == zendnn_any_engine) {
        return "any";
    }
    if (v == zendnn_cpu) {
        return "cpu";
    }
    if (v == zendnn_gpu) {
        return "gpu";
    }
    assert(!"unknown engine_kind");
    return "unknown engine_kind";
}

const char *zendnn_fpmath_mode2str(zendnn_fpmath_mode_t v) {
    if (v == zendnn_fpmath_mode_strict) {
        return "fpmath_mode_strict";
    }
    if (v == zendnn_fpmath_mode_bf16) {
        return "fpmath_mode_bf16";
    }
    if (v == zendnn_fpmath_mode_f16) {
        return "fpmath_mode_f16";
    }
    if (v == zendnn_fpmath_mode_any) {
        return "any";
    }
    assert(!"unknown fpmath_mode");
    return "unknown fpmath_mode";
}

const char *zendnn_scratchpad_mode2str(zendnn_scratchpad_mode_t v) {
    if (v == zendnn_scratchpad_mode_library) {
        return "library";
    }
    if (v == zendnn_scratchpad_mode_user) {
        return "user";
    }
    assert(!"unknown scratchpad_mode");
    return "unknown scratchpad_mode";
}

const char *zendnn_cpu_isa2str(zendnn_cpu_isa_t v) {
    if (v == zendnn_cpu_isa_all) {
        return "cpu_isa_all";
    }
    if (v == zendnn_cpu_isa_sse41) {
        return "cpu_isa_sse41";
    }
    if (v == zendnn_cpu_isa_avx) {
        return "cpu_isa_avx";
    }
    if (v == zendnn_cpu_isa_avx2) {
        return "cpu_isa_avx2";
    }
    if (v == zendnn_cpu_isa_avx512_mic) {
        return "cpu_isa_avx512_mic";
    }
    if (v == zendnn_cpu_isa_avx512_mic_4ops) {
        return "cpu_isa_avx512_mic_4ops";
    }
    if (v == zendnn_cpu_isa_avx512_core) {
        return "cpu_isa_avx512_core";
    }
    if (v == zendnn_cpu_isa_avx512_core_vnni) {
        return "cpu_isa_avx512_core_vnni";
    }
    if (v == zendnn_cpu_isa_avx512_core_bf16) {
        return "cpu_isa_avx512_core_bf16";
    }
    if (v == zendnn_cpu_isa_avx512_core_amx) {
        return "cpu_isa_avx512_core_amx";
    }
    if (v == zendnn_cpu_isa_avx2_vnni) {
        return "cpu_isa_avx2_vnni";
    }
    assert(!"unknown cpu_isa");
    return "unknown cpu_isa";
}

const char *zendnn_cpu_isa_hints2str(zendnn_cpu_isa_hints_t v) {
    if (v == zendnn_cpu_isa_no_hints) {
        return "cpu_isa_no_hints";
    }
    if (v == zendnn_cpu_isa_prefer_ymm) {
        return "cpu_isa_prefer_ymm";
    }
    assert(!"unknown cpu_isa_hints");
    return "unknown cpu_isa_hints";
}


