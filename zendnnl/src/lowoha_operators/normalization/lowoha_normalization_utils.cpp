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

#include "lowoha_normalization_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace normalization {

status_t validate_normalization_inputs(const void *input, const void *output,
        const void *gamma, const void *beta, const void *running_mean,
        const void *running_var, const void *residual,
        const norm_params &params) {
    // Validate pointers
    if (!input) {
        log_error("Normalization: Input pointer is null");
        return status_t::failure;
    }
    if (!output) {
        log_error("Normalization: Output pointer is null");
        return status_t::failure;
    }

    // Validate normalization type
    if (params.norm_type == norm_type_t::NONE) {
        log_error("Normalization: norm_type not specified");
        return status_t::failure;
    }

    // Validate data types
    if (params.src_dt != data_type_t::f32 && params.src_dt != data_type_t::bf16
            && params.src_dt != data_type_t::f16) {
        log_error("Normalization: Unsupported source data type");
        return status_t::failure;
    }
    if (params.dst_dt != data_type_t::f32 && params.dst_dt != data_type_t::bf16
            && params.dst_dt != data_type_t::f16) {
        log_error("Normalization: Unsupported destination data type");
        return status_t::failure;
    }

    // Disallow f16/bf16 cross-mixing between src and dst (matches matmul/embag policy)
    const bool src_is_f16 = (params.src_dt == data_type_t::f16);
    const bool src_is_bf16 = (params.src_dt == data_type_t::bf16);
    const bool dst_is_f16 = (params.dst_dt == data_type_t::f16);
    const bool dst_is_bf16 = (params.dst_dt == data_type_t::bf16);
    if ((src_is_f16 && dst_is_bf16) || (src_is_bf16 && dst_is_f16)) {
        log_error(
                "Normalization: f16/bf16 cross-mixing between src and dst "
                "is not supported");
        return status_t::failure;
    }

    // Validate batch and norm_size
    if (params.batch == 0) {
        log_error("Normalization: batch must be > 0");
        return status_t::failure;
    }
    if (params.norm_size == 0) {
        log_error("Normalization: norm_size must be > 0");
        return status_t::failure;
    }

    // Validate scale (gamma) parameter
    if (params.use_scale) {
        if (!gamma) {
            log_error(
                    "Normalization: use_scale=true but gamma pointer is null");
            return status_t::failure;
        }
        if (params.gamma_dt != data_type_t::f32
                && params.gamma_dt != data_type_t::bf16
                && params.gamma_dt != data_type_t::f16) {
            log_error("Normalization: Unsupported gamma data type (",
                    static_cast<int>(params.gamma_dt),
                    "). Supported: f32, bf16, f16");
            return status_t::failure;
        }
    }

    // Validate shift (beta) parameter
    // Beta is never a shift for the RMS family regardless of use_shift
    // (RMSNormAddRMSNorm reads it as its second scale, validated below)
    if (params.use_shift && params.norm_type != norm_type_t::RMS_NORM
            && params.norm_type != norm_type_t::FUSED_ADD_RMS_NORM
            && params.norm_type != norm_type_t::RMS_NORM_ADD_RMS_NORM) {
        if (!beta) {
            log_error(
                    "Normalization: use_shift=true but beta pointer is null "
                    "(required for ",
                    norm_type_to_str(params.norm_type), ")");
            return status_t::failure;
        }
        if (params.beta_dt != data_type_t::f32
                && params.beta_dt != data_type_t::bf16
                && params.beta_dt != data_type_t::f16) {
            log_error("Normalization: Unsupported beta data type (",
                    static_cast<int>(params.beta_dt),
                    "). Supported: f32, bf16, f16");
            return status_t::failure;
        }
    }

    // FusedAddRMSNorm-specific validations
    if (params.norm_type == norm_type_t::FUSED_ADD_RMS_NORM) {
        if (!residual) {
            log_error(
                    "Normalization: FUSED_ADD_RMS_NORM requires a non-null "
                    "writable residual buffer");
            return status_t::failure;
        }
    }

    // RMSNormAddRMSNorm-specific validations. The residual (dst_dt) is
    // read-modify-written while the output is stored, so the two must not
    // alias; the output may alias the input only when src_dt == dst_dt;
    // `beta` carries the second scale.
    if (params.norm_type == norm_type_t::RMS_NORM_ADD_RMS_NORM) {
        if (!residual) {
            log_error(
                    "Normalization: RMS_NORM_ADD_RMS_NORM requires a non-null "
                    "writable residual buffer");
            return status_t::failure;
        }
        if (residual == output || residual == input) {
            log_error(
                    "Normalization: RMS_NORM_ADD_RMS_NORM residual must not "
                    "alias the input or output buffer");
            return status_t::failure;
        }
        if (output == input && params.src_dt != params.dst_dt) {
            log_error(
                    "Normalization: RMS_NORM_ADD_RMS_NORM with output "
                    "aliasing input requires src_dt == dst_dt");
            return status_t::failure;
        }
        if (params.use_scale) {
            if (!beta) {
                log_error(
                        "Normalization: RMS_NORM_ADD_RMS_NORM with use_scale "
                        "requires beta (the second scale)");
                return status_t::failure;
            }
            if (params.beta_dt != data_type_t::f32
                    && params.beta_dt != data_type_t::bf16
                    && params.beta_dt != data_type_t::f16) {
                log_error(
                        "Normalization: Unsupported second scale (beta) "
                        "data type (",
                        static_cast<int>(params.beta_dt),
                        "). Supported: f32, bf16, f16");
                return status_t::failure;
            }
        }
    }

    // FusedLayerNormAdd-specific validations. The residual is a read-only
    // addend applied to the LayerNorm output (norm-then-add), so it lives in the
    // output domain and its element type must be dst_dt.
    if (params.norm_type == norm_type_t::FUSED_LAYER_NORM_ADD) {
        if (!residual) {
            log_error(
                    "Normalization: FUSED_LAYER_NORM_ADD requires a non-null "
                    "residual buffer");
            return status_t::failure;
        }
        if (residual == output) {
            log_error(
                    "Normalization: FUSED_LAYER_NORM_ADD residual must not "
                    "alias "
                    "the output buffer");
            return status_t::failure;
        }
    }

    // BatchNorm-specific validations
    if (params.norm_type == norm_type_t::BATCH_NORM) {
        if (!running_mean) {
            log_error(
                    "Normalization: BatchNorm inference requires running_mean "
                    "(pre-computed from training)");
            return status_t::failure;
        }
        if (!running_var) {
            log_error(
                    "Normalization: BatchNorm inference requires running_var "
                    "(pre-computed from training)");
            return status_t::failure;
        }
    }

    // Epsilon validation
    if (params.epsilon <= 0.0f) {
        log_error("Normalization: epsilon must be > 0, got ", params.epsilon);
        return status_t::failure;
    }

    return status_t::success;
}

status_t validate_gated_norm_params(const norm_params &params) {
    const gated_norm_params &gp = params.gated;
    const bool gated = params.norm_type == norm_type_t::FUSED_GATED_LAYER_NORM;
    if (!gated) {
        if (params.src_stride || params.dst_stride || gp.gate || gp.gate_stride
                || gp.gate_out || gp.gate_out_stride || gp.src_out
                || gp.src_out_stride || gp.gate_silu) {
            log_error(
                    "Normalization: row strides and norm_params::gated are "
                    "only supported by FUSED_GATED_LAYER_NORM");
            return status_t::failure;
        }
        return status_t::success;
    }
    if (!gp.gate) {
        log_error(
                "Normalization: FUSED_GATED_LAYER_NORM requires a non-null "
                "gate");
        return status_t::failure;
    }
    if (gp.gate_dt != data_type_t::f32 && gp.gate_dt != data_type_t::bf16
            && gp.gate_dt != data_type_t::f16) {
        log_error("Normalization: Unsupported gate data type (",
                static_cast<int>(gp.gate_dt), "). Supported: f32, bf16, f16");
        return status_t::failure;
    }
    for (uint64_t s : {params.src_stride, params.dst_stride, gp.gate_stride,
                 gp.gate_out_stride, gp.src_out_stride}) {
        if (s != 0 && s < params.norm_size) {
            log_error("Normalization: row stride ", s,
                    " is smaller than norm_size ", params.norm_size);
            return status_t::failure;
        }
    }
    return status_t::success;
}

std::string norm_type_to_str(norm_type_t type) {
    switch (type) {
        case norm_type_t::LAYER_NORM: return "LayerNorm";
        case norm_type_t::BATCH_NORM: return "BatchNorm";
        case norm_type_t::RMS_NORM: return "RMSNorm";
        case norm_type_t::FUSED_ADD_RMS_NORM: return "FusedAddRMSNorm";
        case norm_type_t::FUSED_LAYER_NORM_ADD: return "FusedLayerNormAdd";
        case norm_type_t::FUSED_GATED_LAYER_NORM: return "FusedGatedLayerNorm";
        case norm_type_t::RMS_NORM_ADD_RMS_NORM: return "RMSNormAddRMSNorm";
        default: return "Unknown";
    }
}

const char *algo_to_string(norm_algo_t algo) {
    switch (algo) {
        case norm_algo_t::none: return "none";
        case norm_algo_t::dynamic_dispatch: return "dynamic_dispatch";
        case norm_algo_t::reference: return "reference";
        default: return "unknown";
    }
}

norm_algo_t algo_select(norm_params &params) {
    norm_algo_t algo = params.algorithm == norm_algo_t::none
            ? norm_algo_t::dynamic_dispatch
            : params.algorithm;

    if (algo != norm_algo_t::dynamic_dispatch
            && algo != norm_algo_t::reference) {
        return params.algorithm;
    }

    params.algorithm = algo;
    return algo;
}

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl
