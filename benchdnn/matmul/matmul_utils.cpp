/********************************************************************************
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

#include "matmul_utils.hpp"

#include <cctype>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace zendnnl {
namespace benchdnn {
namespace matmul {

static bool is_w4a8_benchdnn_config(const MatmulConfig &cfg) {
    return cfg.src_dynamic_quant && cfg.dt.size() >= 3
            && cfg.dt[0] == data_type_t::bf16 && cfg.dt[1] == data_type_t::s4
            && cfg.dt[2] == data_type_t::bf16;
}

void normalize_w4a8_quant_config(MatmulConfig &cfg) {
    if (!is_w4a8_benchdnn_config(cfg)) {
        if (cfg.dt.size() >= 2
                && (cfg.dt[1] == data_type_t::s4 || cfg.dt[1] == data_type_t::s8
                        || cfg.dt[1] == data_type_t::u4)
                && cfg.scale_granularity == "none") {
            cfg.scale_granularity = "channel";
            commonlog_warning(
                    "No weight scale granularity specified. Defaulting to "
                    "'per-channel'.");
        }
        return;
    }

    if (cfg.scale_granularity != "group") {
        cfg.scale_granularity = "group";
        commonlog_warning(
                "W4A8 requires per-group weight scales; using per-group.");
    }
    if (cfg.group_size == 0) { cfg.group_size = cfg.k; }
    if (cfg.src_scale_granularity == "per-group" && cfg.src_group_size == 0) {
        cfg.src_group_size = cfg.group_size;
    }
}

// Enforces the granularity/group-size pairing shared by weight and src scales:
//   per-group  -> group size must be non-zero
//   per-token / per-channel -> group size must be zero
// Granularity "none"/"per-tensor" imposes no group-size constraint.
static bool check_granularity_group_size(const std::string &granularity,
        uint64_t group_size, const char *what, const std::string &ctx) {
    const bool is_group
            = (granularity == "group" || granularity == "per-group");
    const bool is_token
            = (granularity == "channel" || granularity == "per-token");
    if (is_group && group_size == 0) {
        commonlog_error(ctx, ": ", what, " granularity '", granularity,
                "' requires a non-zero group size.");
        return false;
    }
    if (is_token && group_size != 0) {
        commonlog_error(ctx, ": ", what, " granularity '", granularity,
                "' requires group size 0 (got ", group_size, ").");
        return false;
    }
    return true;
}

// Strip inline '#' comments from an input line (everything from '#' onward).
static void strip_inline_hash_comment(std::string &line) {
    const std::size_t hashPos = line.find('#');
    if (hashPos != std::string::npos) { line.erase(hashPos); }
}

// Normalize the src-scale granularity/group-size for a config that carries a
// source scale (dynamic or static int8/int4). Src and weight granularity are
// independent, so an explicit per-token/per-group src granularity is honored;
// only an unset/invalid one defaults to mirroring the weight granularity. A
// per-group src with no group size falls back to the weight group size (or K).
static void couple_src_scale_to_weight(MatmulConfig &cfg) {
    const bool wei_per_group = (cfg.scale_granularity == "group");
    if (cfg.src_scale_granularity != "per-group"
            && cfg.src_scale_granularity != "per-token") {
        cfg.src_scale_granularity = wei_per_group ? "per-group" : "per-token";
    }
    if (cfg.src_scale_granularity == "per-group") {
        if (cfg.src_group_size == 0) {
            cfg.src_group_size = cfg.group_size != 0 ? cfg.group_size : cfg.k;
        }
    } else {
        cfg.src_group_size = 0;
    }
}

// Keep src/weight group sizes in sync only for per-group src scales.
// Per-token/per-tensor src must keep src_group_size=0 even when weights use
// per-group scales (W4A8).
static void sync_src_weight_group_sizes(MatmulConfig &cfg) {
    const bool src_per_group = (cfg.src_scale_granularity == "per-group"
            || cfg.src_scale_granularity == "group");
    if (src_per_group) {
        if (cfg.src_group_size != cfg.group_size) {
            if (cfg.src_group_size == 0) {
                cfg.src_group_size = cfg.group_size;
            } else if (cfg.group_size == 0) {
                cfg.group_size = cfg.src_group_size;
            } else {
                commonlog_warning("src_group_size=", cfg.src_group_size,
                        " differs from weight group_size=", cfg.group_size,
                        ". Forcing both to ", cfg.group_size, ".");
                cfg.src_group_size = cfg.group_size;
            }
        }
    } else {
        cfg.src_group_size = 0;
    }
}

// Validates/normalizes a parsed row against its weight dtype. The row parser is
// positional and defaults missing tail fields, so a quantized row missing quant
// metadata would otherwise parse "successfully" then crash at kernel dispatch.
// Returns false to skip the row; may adjust cfg (forces bf16 src/dst, src
// scale dtype to the weight scale dtype, and a default src granularity when
// the row leaves it unset). bf16->bf16:bf16:bf16, s8->bf16:s8:bf16,
// s4->bf16:s4:bf16.
static bool validate_dtype_fields(MatmulConfig &cfg, const std::string &line) {
    if (cfg.dt.size() < 3) {
        commonlog_error(
                "Expected 3 data types (in:weights:out) for row: ", line);
        return false;
    }

    const data_type_t src_dt = cfg.dt[0];
    const data_type_t wei_dt = cfg.dt[1];
    const bool is_int8 = (wei_dt == data_type_t::s8);
    const bool is_int4
            = (wei_dt == data_type_t::s4 || wei_dt == data_type_t::u4);
    const bool is_bf16 = (wei_dt == data_type_t::bf16);

    // Static int8 activation quant: the source is already integer (s8) with a
    // precomputed (static) scale, as opposed to INT8 dynamic quant where a bf16/f32
    // source is quantized to s8 at runtime. It is detected from the source dtype
    // so we neither rewrite the activations to bf16 nor require dynamic quant.
    // Note: u8 (asymmetric) is not currently supported in benchdnn static int8;
    // u8 src/dst rows are normalized below (see validate_dtype_fields).
    const bool is_static_int8 = is_int8 && (src_dt == data_type_t::s8);

    // u8 (asymmetric) is not currently exercised in benchdnn matmul; normalize so
    // rows still run with the nearest supported dtype triple.
    if (is_int8 && src_dt == data_type_t::u8) {
        commonlog_warning("Row '", line,
                "' u8 src is not currently supported in benchdnn matmul; "
                "forcing src to bf16.");
        cfg.dt[0] = data_type_t::bf16;
    }

    // Enforce the canonical dtype triple: activations and output are bf16 for the
    // bf16 / dynamic-int8 / int4 weight cases. Populate (and warn) rather than
    // fail so a row that only got the weight dtype right still runs with the
    // intended config. Static int8 is excluded: it keeps its integer source and
    // its chosen output dtype (s8 or bf16).
    if ((is_bf16 || is_int8 || is_int4) && !is_static_int8) {
        if (cfg.dt[0] != data_type_t::bf16 || cfg.dt[2] != data_type_t::bf16) {
            commonlog_warning("Row '", line,
                    "' expected dtype 'bf16:", datatypeToStr(wei_dt),
                    ":bf16' but got '", datatypeToStr(cfg.dt[0]), ":",
                    datatypeToStr(wei_dt), ":", datatypeToStr(cfg.dt[2]),
                    "'. Forcing src and dst to bf16.");
            cfg.dt[0] = data_type_t::bf16;
            cfg.dt[2] = data_type_t::bf16;
        }
    }

    const std::string dt_str = datatypeToStr(cfg.dt[0]) + ":"
            + datatypeToStr(wei_dt) + ":" + datatypeToStr(cfg.dt[2]);

    if (is_static_int8) {
        // Static INT8: integer (s8) source with a precomputed per-tensor scale
        // (created by the tensor factory). No runtime dynamic quant. The output may
        // be integer (s8) or dequantized (bf16/f32). u8 is not currently supported
        // in benchdnn static int8 (see u8 normalization above / dst check below).
        // Weight scales are still required.
        if (cfg.src_dynamic_quant) {
            commonlog_warning("Row '", line, "' uses a static int8 source (",
                    dt_str,
                    ") but src_dynamic_quant is enabled; static and dynamic "
                    "quant are "
                    "mutually exclusive. Forcing src_dynamic_quant to false.");
            cfg.src_dynamic_quant = false;
        }
        if (cfg.dt[2] != data_type_t::s8 && cfg.dt[2] != data_type_t::bf16
                && cfg.dt[2] != data_type_t::f32) {
            commonlog_warning("Row '", line, "' static int8 dst dtype '",
                    datatypeToStr(cfg.dt[2]),
                    "' is unsupported (use s8, bf16 or f32; u8 is not "
                    "currently supported). "
                    "Forcing dst to bf16.");
            cfg.dt[2] = data_type_t::bf16;
        }
        if (cfg.scale_granularity != "group"
                && cfg.scale_granularity != "channel") {
            commonlog_error("Row '", line,
                    "' static int8 weights require a weight scale "
                    "granularity (group|channel), got '",
                    cfg.scale_granularity, "'.");
            return false;
        }
        // The static-quant scale layout depends on the output dtype:
        //   * Integer (s8) output applies the weight dequant scale as a
        //     per-channel post-op (length N) and only supports a per-tensor static
        //     source scale (dlp: "Post_op.scale PER_CHANNEL requires
        //     scale_factor_len == n"). Per-group weight/src is not expressible.
        //   * Dequantized (bf16/f32) output supports per-group/per-channel weights
        //     and per-group/per-token static source scales.
        if (cfg.dt[2] == data_type_t::s8) {
            if (cfg.scale_granularity != "channel") {
                commonlog_warning("Row '", line,
                        "' static int8 with integer output (", dt_str,
                        ") only supports per-channel weight scales; forcing "
                        "weight "
                        "granularity to per-channel.");
                cfg.scale_granularity = "channel";
            }
            cfg.group_size = 0;
            cfg.src_scale_granularity = "per-tensor";
            cfg.src_group_size = 0;
        } else {
            if (cfg.scale_granularity == "channel") { cfg.group_size = 0; }
            couple_src_scale_to_weight(cfg);
        }
        // Match the weight scale dtype (kernel requires src and weight scale dtypes
        // to be equal).
        if (cfg.src_scale_dt != cfg.scale_dt) {
            cfg.src_scale_dt = cfg.scale_dt;
        }
        if (!check_granularity_group_size(cfg.src_scale_granularity,
                    cfg.src_group_size, "src scale", line)) {
            return false;
        }
    } else if (is_int8) {
        // INT8 dynamic quant (s8 weights). The integer GEMM needs the
        // activation pre/post-quant metadata, which is only produced when
        // src_dynamic_quant is on.
        if (!cfg.src_dynamic_quant) {
            commonlog_error("Row '", line, "' uses int8 weights (", dt_str,
                    ") but src_dynamic_quant is not enabled. int8 requires "
                    "dynamic source "
                    "quant; append the tail fields: "
                    "src_dynamic_quant=true, src_scale_granularity "
                    "(per-token|per-group), "
                    "src_group_size, src_scale_dt.");
            return false;
        }
        if (cfg.scale_granularity != "group"
                && cfg.scale_granularity != "channel") {
            commonlog_error("Row '", line,
                    "' int8 weights require a weight scale granularity "
                    "(group|channel), got '",
                    cfg.scale_granularity, "'.");
            return false;
        }
        // INT8 dynamic (s8 weights). Src and weight granularity are
        // independent, matching validate_matmul_direct_inputs:
        //   per-channel wei -> src must be per-token
        //   per-group wei   -> src may be per-group (legacy default) OR
        //                      explicit per-token (src_group_size=0)
        if (cfg.scale_granularity == "channel") {
            if (cfg.src_scale_granularity != "per-token") {
                commonlog_warning("Row '", line, "' weight granularity '",
                        cfg.scale_granularity,
                        "' requires src_scale_granularity 'per-token' but "
                        "got '",
                        cfg.src_scale_granularity,
                        "'. Forcing src to 'per-token'.");
                cfg.src_scale_granularity = "per-token";
            }
            cfg.group_size = 0;
            cfg.src_group_size = 0;
        } else if (cfg.src_scale_granularity == "per-token") {
            // Explicit mixed pairing: per-token src + per-group wei.
            cfg.src_group_size = 0;
        } else {
            if (cfg.src_scale_granularity != "per-group") {
                commonlog_warning("Row '", line, "' weight granularity '",
                        cfg.scale_granularity,
                        "' defaults src to 'per-group' (got '",
                        cfg.src_scale_granularity, "').");
                cfg.src_scale_granularity = "per-group";
            }
            if (cfg.src_group_size == 0) {
                cfg.src_group_size = cfg.group_size;
            }
        }
        // The GEMM kernel requires the src (A) and weight (B) scale dtypes
        // to match (dlp_gemm_post_ops.c: "A and B scale factor type
        // mismatch"). Force the src scale dtype to the weight scale dtype.
        if (cfg.src_scale_dt != cfg.scale_dt) {
            commonlog_warning("Row '", line, "' src scale dtype (",
                    datatypeToStr(cfg.src_scale_dt),
                    ") must match weight scale dtype (",
                    datatypeToStr(cfg.scale_dt),
                    "); forcing src scale dtype to ",
                    datatypeToStr(cfg.scale_dt), ".");
            cfg.src_scale_dt = cfg.scale_dt;
        }
    } else if (is_int4) {
        // Weight-only (or W4A8) quant. Weight scales are always needed; the shared
        // group-size check below rejects 'group' with a zero group size.
        if (cfg.scale_granularity == "none") {
            commonlog_error("Row '", line,
                    "' int4 weights require a weight scale granularity "
                    "(group|channel), got 'none'.");
            return false;
        }
    } else {
        // Plain float weights (bf16/f32): there is no per-group/per-token quant, so
        // any quant/scale tail fields after warmup_iters are meaningless. Silently
        // ignore them by resetting to the non-quantized defaults.
        cfg.src_dynamic_quant = false;
        cfg.src_scale_granularity = "per-tensor";
        cfg.src_group_size = 0;
        cfg.src_scale_dt = data_type_t::f32;
        cfg.scale_granularity = "none";
        cfg.group_size = 0;
        cfg.scale_dt = data_type_t::f32;
    }

    if (!check_granularity_group_size(
                cfg.scale_granularity, cfg.group_size, "weight scale", line)) {
        return false;
    }
    if (cfg.src_dynamic_quant
            && !check_granularity_group_size(cfg.src_scale_granularity,
                    cfg.src_group_size, "src scale", line)) {
        return false;
    }
    return true;
}

void inputFileParser(std::ifstream &infile, std::vector<MatmulConfig> &configs,
        bool &isPipeline, const global_options &options) {
    std::string line;

    // Parse each line of the input file into a MatmulConfig object
    while (std::getline(infile, line)) {
        if (line.empty()) { continue; }

        // Strip inline '#' comments and skip comment-only lines. This lets input
        // files carry documentation headers, section separators, and per-line
        // annotations (e.g. "# Llama-3.1_8B") without affecting parsing.
        strip_inline_hash_comment(line);
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        // Split the line into fields and validate.
        //
        // Field layout:
        //   mandatory: ndims-prefix (bs?, m, k) + 1 (n) + MATMUL_EXTRA_INPUT_FIELD_COUNT (15)
        //   optional positional, in order:
        //     warmup_iters,                                                        (1 field)
        //     src_dynamic_quant, src_scale_granularity, src_group_size, src_scale_dt (4 fields)
        //
        // Older inputs without the 4 dyn-quant fields keep working unchanged.
        // The optional tail is bounded; anything past it is unknown to the parser
        // and likely indicates a malformed line (e.g. a results CSV mistakenly
        // fed back in as input). We warn rather than fail so existing well-formed
        // inputs with stray trailing commas keep parsing.
        constexpr int MATMUL_OPTIONAL_INPUT_FIELD_COUNT = 5;
        auto fields = split(line, ',');
        const int mandatory_cnt
                = options.ndims + 1 + MATMUL_EXTRA_INPUT_FIELD_COUNT;
        const int max_cnt = mandatory_cnt + MATMUL_OPTIONAL_INPUT_FIELD_COUNT;
        if (static_cast<int>(fields.size()) < mandatory_cnt) {
            commonlog_error("Invalid line (expected at least ", mandatory_cnt,
                    " fields): [", (options.ndims > 2) ? "bs, " : "",
                    "m, k, n, iterations, "
                    "input_dtype:weights_dtype:output_dtype, isBiasEnabled, "
                    "bias_dtype, postOp, postOp_dtype, ",
                    "kernel name, isWeightsConst, isTransA, isTransB, alpha, "
                    "beta, ",
                    "weight_scale_granularity, weight_group_size, "
                    "weight_scale_dt, warmup_iters (optional), ",
                    "src_dynamic_quant (optional), src_scale_granularity "
                    "(optional), src_group_size (optional), src_scale_dt "
                    "(optional)]");
            continue;
        }
        if (static_cast<int>(fields.size()) > max_cnt) {
            commonlog_warning("Line has ", fields.size(),
                    " fields but parser understands at most ", max_cnt, " (",
                    mandatory_cnt, " mandatory + ",
                    MATMUL_OPTIONAL_INPUT_FIELD_COUNT,
                    " optional). Extra trailing fields will be ignored; verify "
                    "the input "
                    "is not a results CSV or otherwise malformed.");
        }
        MatmulConfig cfg;
        try {
            int id = 0;
            if (options.ndims > 2) {
                if (fields[id].empty() || std::stoi(fields[id]) <= 0) {
                    commonlog_error(
                            "BS value cannot be empty or <= 0. Please provide "
                            "a valid number.");
                    continue;
                }
                cfg.bs = std::stoi(fields[id++]);
            } else {
                cfg.bs = 1;
            }
            if (fields[id].empty() || std::stoi(fields[id]) <= 0) {
                commonlog_error(
                        "M value cannot be empty or <= 0. Please provide a "
                        "valid number.");
                continue;
            }
            cfg.m = std::stoi(fields[id++]);
            if (fields[id].empty() || std::stoi(fields[id]) <= 0) {
                commonlog_error(
                        "K value cannot be empty or <= 0. Please provide a "
                        "valid number.");
                continue;
            }
            cfg.k = std::stoi(fields[id++]);
            if (fields[id].empty()) {
                commonlog_error(
                        "N values cannot be empty. Please provide a valid "
                        "number.");
                continue;
            }
            auto n_values = split(fields[id++], ':');
            for (const auto &n : n_values) {
                if (n.empty() || std::stoi(n) <= 0) {
                    commonlog_error(
                            "One of the n values is empty or <= 0. Please "
                            "provide a valid value.");
                    continue;
                }
                cfg.n_values.push_back(std::stoi(n));
            }
            // Set isPipeline to true if more than one n value is present
            isPipeline = (n_values.size() > 1) ? true : isPipeline;
            if (fields[id].empty()) {
                commonlog_error(
                        "Field for iterations is empty. Please provide a "
                        "value.");
                continue;
            }
            cfg.iters = std::stoi(fields[id++]);
            // Parse data types (input:weights:output)
            auto dt = split(fields[id++], ':');
            cfg.provided.dt = fields[id - 1].size() > 0;
            if (fields[id - 1].size() > 0) {
                auto i = 0;
                for (; i < dt.size(); i++) {
                    cfg.dt.push_back(strToDatatype(dt[i]));
                }
                for (; i < 3; i++) {
                    cfg.dt.push_back(data_type_t::f32);
                }
                if (dt.size() < 3) {
                    commonlog_warning(
                            "Less than 3 data types specified. Defaulting "
                            "missing types to f32.");
                }
            } else {
                cfg.dt.push_back(data_type_t::f32);
                cfg.dt.push_back(data_type_t::f32);
                cfg.dt.push_back(data_type_t::f32);
                commonlog_warning(
                        "No data types specified. Defaulting all to f32.");
            }
            // Parse bias flag and bias data type
            if (fields[id].empty()) {
                commonlog_error(
                        "Field for isBiasEnabled is empty. Please provide a "
                        "value.");
                continue;
            }
            std::string bias_flag = fields[id];
            std::transform(bias_flag.begin(), bias_flag.end(),
                    bias_flag.begin(), ::tolower);
            if (bias_flag == "true" || bias_flag == "1") {
                cfg.isBiasEnabled = true;
            } else if (bias_flag == "false" || bias_flag == "0") {
                cfg.isBiasEnabled = false;
            } else {
                commonlog_error(
                        "Invalid value for isBiasEnabled. Use true/false or "
                        "1/0.");
                continue;
            }
            id++;
            if (cfg.isBiasEnabled) {
                if (!fields[id++].empty()) {
                    cfg.bias_dt = strToDatatype(fields[id - 1]);
                } else {
                    commonlog_warning(
                            "No data type specified for bias. Defaulting it to "
                            "f32.");
                    cfg.bias_dt = data_type_t::f32;
                }
            } else {
                id++; // Skip bias data type field if bias is not enabled
            }
            // Parse post-operations (e.g., relu, gelu, binary ops)
            auto postOps = split(fields[id++], ':');
            auto binary_post_op_pos = 0;
            for (auto i = 0; i < postOps.size(); i++) {
                if (!postOps[i].empty()) {
                    cfg.post_ops.push_back(strToPostOps(postOps[i]));
                    // Track positions of binary post-operations
                    if (postOps[i].find("binary_") == 0) {
                        cfg.binary_post_ops_pos.push_back(binary_post_op_pos);
                    }
                    binary_post_op_pos++;
                }
            }

            if (binary_post_op_pos > 0) {
                if (fields[id].empty()) {
                    commonlog_warning(
                            "No postOp_dtype specified for binary "
                            "post-operation. Defaulting it to f32.");
                    cfg.post_op_dt = data_type_t::f32;
                } else {
                    cfg.post_op_dt = strToDatatype(fields[id]);
                }
            }
            id++;

            zendnnl::common::matmul_config_t &matmul_config
                    = zendnnl::common::matmul_config_t::instance();
            int32_t algo_ = options.ndims > 2 ? matmul_config.get_bmm_algo()
                                              : matmul_config.get_algo();
            matmul_algo_t algo = static_cast<matmul_algo_t>(algo_);
            if (algo == matmul_algo_t::none) {
                // Parse kernel name
                if (fields[id].empty()) {
                    auto kernel_name = options.ndims > 2 ? "aocl_dlp"
                                                         : "aocl_dlp_blocked";
                    commonlog_warning(
                            "No kernel name specified. Defaulting to '",
                            kernel_name, "'.");
                    cfg.kernel_name = kernel_name;
                } else {
                    cfg.kernel_name = fields[id];
                    if (!validateMatmulKernelName(cfg.kernel_name)) {
                        auto kernel_name = options.ndims > 2
                                ? "aocl_dlp"
                                : "aocl_dlp_blocked";
                        commonlog_warning("Unknown kernel name '",
                                cfg.kernel_name,
                                "'. Supported: aocl_dlp_blocked, "
                                "onednn_blocked, libxsmm_blocked, aocl_dlp, "
                                "onednn, libxsmm, "
                                "batched_sgemm, auto, dynamic_dispatch, "
                                "reference. Using '",
                                kernel_name, "' instead.");
                        cfg.kernel_name = kernel_name;
                    }
                }
            } else {
                cfg.kernel_name = algoToStr(algo);
            }
            // A forced algo or a non-empty kernel field both count as user-provided.
            cfg.provided.kernel
                    = (algo != matmul_algo_t::none) || !fields[id].empty();
            id++;

            if (fields[id].empty()) {
                cfg.is_weights_const = options.ndims > 2 ? false : true;
            } else {
                std::string is_weights_const = fields[id];
                std::transform(is_weights_const.begin(), is_weights_const.end(),
                        is_weights_const.begin(), ::tolower);
                if (is_weights_const == "true" || is_weights_const == "1") {
                    cfg.is_weights_const = true;
                } else if (is_weights_const == "false"
                        || is_weights_const == "0") {
                    cfg.is_weights_const = false;
                } else {
                    commonlog_error(
                            "Invalid value for is_weights_const. Use "
                            "true/false or 1/0.");
                    continue;
                }
            }
            id++;

            std::string transA_flag = fields[id];
            std::transform(transA_flag.begin(), transA_flag.end(),
                    transA_flag.begin(), ::tolower);
            if (transA_flag == "true" || transA_flag == "1") {
                cfg.isTransA = true;
            } else if (transA_flag == "false" || transA_flag == "0") {
                cfg.isTransA = false;
            } else {
                commonlog_error(
                        "Invalid value for isTransA. Use true/false or 1/0.");
                continue;
            }
            id++;

            std::string transB_flag = fields[id];
            std::transform(transB_flag.begin(), transB_flag.end(),
                    transB_flag.begin(), ::tolower);
            if (transB_flag == "true" || transB_flag == "1") {
                cfg.isTransB = true;
            } else if (transB_flag == "false" || transB_flag == "0") {
                cfg.isTransB = false;
            } else {
                commonlog_error(
                        "Invalid value for isTransB. Use true/false or 1/0.");
                continue;
            }
            id++;
            // Parse alpha and beta scaling factors (default: alpha=1.0, beta=0.0)
            cfg.alpha = fields[id].empty() ? 1.0f : std::stof(fields[id]);
            id++;
            cfg.beta = fields[id].empty() ? 0.0f : std::stof(fields[id]);
            id++;
            if (cfg.dt[1] == data_type_t::s4 || cfg.dt[1] == data_type_t::s8
                    || cfg.dt[1] == data_type_t::u4) {
                cfg.provided.wei_scale = !fields[id].empty();
                if (!fields[id].empty()) {
                    std::string scale_gran = fields[id];
                    std::transform(scale_gran.begin(), scale_gran.end(),
                            scale_gran.begin(), ::tolower);
                    if (scale_gran == "per-channel"
                            || scale_gran == "channel") {
                        cfg.scale_granularity = "channel";
                    } else if (scale_gran == "per-group"
                            || scale_gran == "group") {
                        cfg.scale_granularity = "group";
                    } else if (scale_gran == "per-tensor"
                            || scale_gran == "tensor") {
                        cfg.scale_granularity = "tensor";
                    } else if (scale_gran == "none") {
                        cfg.scale_granularity = "none";
                    } else {
                        cfg.scale_granularity = "channel";
                        commonlog_warning(
                                "Invalid value for weight scale granularity. "
                                "Defaulting to 'per-channel'.");
                    }
                } else {
                    // Default to per-channel if not specified
                    cfg.scale_granularity = "channel";
                    commonlog_warning(
                            "No weight scale granularity specified. Defaulting "
                            "to 'per-channel'.");
                }
                id++;
                cfg.group_size
                        = fields[id].empty() ? 0 : std::stoul(fields[id]);
                id++;
                // Defaulting scale data type to f32 if not specified
                cfg.provided.wei_scale_dt = !fields[id].empty();
                cfg.scale_dt = fields[id].empty()
                        ? zendnnl::common::data_type_t::f32
                        : strToDatatype(fields[id]);
                id++;
            } else {
                cfg.scale_granularity = "none";
                cfg.group_size = 0;
                cfg.scale_dt = zendnnl::common::data_type_t::f32;
                id += 3;
            }
            // Parse warmup iterations if the next field is empty (explicit
            // placeholder) or looks numeric. If it's non-empty and non-numeric
            // (e.g. "true"), the user likely skipped the warmup placeholder and
            // went straight to src_dynamic_quant; in that case default warmup and
            // leave id alone so the dyn-quant parser picks the field up.
            auto looks_signed_int = [](const std::string &s) -> bool {
                if (s.empty()) { return false; }
                size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
                if (i == s.size()) { return false; }
                return std::all_of(s.begin() + i, s.end(), ::isdigit);
            };
            if (id < fields.size() && fields[id].empty()) {
                cfg.warmup_iters = 0.2 * cfg.iters;
                id++;
            } else if (id < fields.size() && looks_signed_int(fields[id])) {
                cfg.warmup_iters = std::stoi(fields[id]);
                id++;
            } else {
                cfg.warmup_iters = 0.2 * cfg.iters;
                // Do not advance id: the field (if any) is not a warmup value; let
                // src_dynamic_quant parsing below consume it.
            }

            // Optional dynamic-source-quant fields:
            //   [src_dynamic_quant, src_scale_granularity, src_group_size, src_scale_dt]
            cfg.src_dynamic_quant = false;
            cfg.src_scale_granularity = "per-tensor";
            cfg.src_group_size = 0;
            cfg.src_scale_dt = zendnnl::common::data_type_t::f32;

            if (id < fields.size() && !(fields[id].empty())) {
                std::string dq = fields[id];
                std::transform(dq.begin(), dq.end(), dq.begin(), ::tolower);
                if (dq == "true" || dq == "1") {
                    cfg.src_dynamic_quant = true;
                } else if (dq == "false" || dq == "0") {
                    cfg.src_dynamic_quant = false;
                } else {
                    commonlog_warning(
                            "Invalid value for src_dynamic_quant. Defaulting "
                            "to 'false'.");
                }
            }
            id++;

            // The src scale granularity is the source of truth for the src-quant
            // block: when present, src_dynamic_quant and src_group_size are treated
            // as file-provided too (they sit alongside it in the row).
            cfg.provided.src_scale
                    = (id < fields.size() && !(fields[id].empty()));
            if (id < fields.size() && !(fields[id].empty())) {
                std::string gran = fields[id];
                std::transform(
                        gran.begin(), gran.end(), gran.begin(), ::tolower);
                if (gran == "per-tensor" || gran == "tensor") {
                    cfg.src_scale_granularity = "per-tensor";
                } else if (gran == "per-token" || gran == "token") {
                    cfg.src_scale_granularity = "per-token";
                } else if (gran == "per-group" || gran == "group") {
                    cfg.src_scale_granularity = "per-group";
                } else {
                    cfg.src_scale_granularity = "per-tensor";
                    commonlog_warning("Invalid src_scale_granularity '",
                            fields[id], "'. Defaulting to 'per-tensor'.");
                }
            }
            id++;

            if (id < fields.size() && !(fields[id].empty())) {
                cfg.src_group_size = std::stoul(fields[id]);
            }
            id++;

            if (id < fields.size() && !(fields[id].empty())) {
                cfg.provided.src_scale_dt = true;
                cfg.src_scale_dt = strToDatatype(fields[id]);
            }

            sync_src_weight_group_sizes(cfg);

            if (!validate_dtype_fields(cfg, line)) { continue; }
            normalize_w4a8_quant_config(cfg);
            configs.push_back(cfg);
        } catch (const std::exception &e) {
            commonlog_error(e.what());
            continue;
        }
    }
}

void inputModelFileParser(std::ifstream &infile,
        std::vector<MatmulConfig> &configs, bool &isPipeline,
        const global_options &options) {
    std::string line;

    // Parse each line of the input file into a MatmulConfig object
    while (std::getline(infile, line)) {
        if (line.empty()) { continue; }

        // Strip inline '#' comments and skip comment-only lines. This lets input
        // files carry documentation headers, section separators, and per-line
        // annotations (e.g. "# Llama-3.1_8B") without affecting parsing.
        strip_inline_hash_comment(line);
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        // Split the line into fields and validate
        auto fields = split(line, ',');
        int fields_size = fields.size();
        int expected_fields_cnt
                = options.ndims + 1; // modelname, [bs], [m], k, n
        if (fields_size > 2) {
            auto to_lower = [](const std::string &s) {
                std::string out = s;
                std::transform(out.begin(), out.end(), out.begin(), ::tolower);
                return out;
            };
            auto is_bool_str = [](const std::string &s) {
                return s == "true" || s == "1" || s == "false" || s == "0";
            };
            auto is_number_str = [](const std::string &s) {
                return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
            };
            std::string last = to_lower(fields[fields_size - 1]);
            std::string prev = (fields_size > 1) ? fields[fields_size - 2] : "";

            if (is_bool_str(last)) {
                // If previous is not a number, treat as trailing [postOp, isBiasEnabled] and ignore both
                if (!is_number_str(prev)) {
                    fields_size -= 2;
                } else if (!(last == "1" || last == "0")) {
                    // If last is not 1/0 and previous is a number, error
                    commonlog_error("Invalid input: ", line, " (expected ",
                            expected_fields_cnt, " fields): <", "ModelName, ",
                            (options.ndims > 2) ? "bs, " : "",
                            "M, K, N, [postOp, isBiasEnabled]>");
                    continue;
                }
                // else: last is 1/0 and previous is a number, do nothing
            } else if (!is_number_str(last)) {
                // If last is not a number, treat as trailing postOp and ignore it
                fields_size--;
            }
            // else: last is a number, do nothing
        }

        if (fields_size < expected_fields_cnt) {
            commonlog_error("Invalid input: ", line, " (expected ",
                    expected_fields_cnt, " fields): <", "ModelName, ",
                    (options.ndims > 2) ? "bs, " : "",
                    "M, K, N, [postOp, isBiasEnabled]>");
            continue;
        }
        MatmulConfig cfg;
        try {
            int id = 0;
            cfg.modelName = fields[id++];
            cfg.bs = 1;
            // Handle different field counts based on ndims
            if (fields_size == 3) {
                if (options.sweep_enabled) {
                    // Placeholder; expand_matmul_sweep() assigns each M from --m_sweep.
                    cfg.m = 1;
                } else if (options.m <= 0) {
                    commonlog_error(
                            "M value cannot be <= 0. Please provide a valid "
                            "number.");
                    continue;
                } else {
                    cfg.m = options.m;
                }
            } else if (fields_size == 4) {
                if (options.ndims != 2) {
                    commonlog_error(
                            "Ensure to provide the correct number of dims and "
                            "specify BS (batch size) for BMM cases. Input: ",
                            line);
                    continue;
                }
                if (fields[id].empty() || std::stoi(fields[id]) <= 0) {
                    commonlog_error(
                            "M value cannot be empty or <= 0. Please provide a "
                            "valid number.");
                    continue;
                } else {
                    cfg.m = std::stoi(fields[id++]);
                }
            } else if (fields_size == 5) {
                if (options.ndims <= 2) {
                    commonlog_error(
                            "Ensure to provide the correct number of dims and "
                            "specify BS (batch size) for BMM cases. Input: ",
                            line);
                    continue;
                }
                if (fields[id].empty()) {
                    commonlog_error(
                            "Field for bs is empty. Please provide a value.");
                    continue;
                } else {
                    cfg.bs = std::stoi(fields[id++]);
                }
                if (fields[id].empty()) {
                    commonlog_error(
                            "Field for m is empty. Please provide a value.");
                    continue;
                } else {
                    cfg.m = std::stoi(fields[id++]);
                }
            }
            if (fields[id].empty()) {
                commonlog_error(
                        "Field for k is empty. Please provide a value.");
                continue;
            } else {
                cfg.k = std::stoi(fields[id++]);
            }
            if (fields[id].empty()) {
                commonlog_error(
                        "Field for n is empty. Please provide a value.");
                continue;
            }
            auto n_values = split(fields[id++], ':');
            for (const auto &n : n_values) {
                if (n.empty()) {
                    commonlog_error(
                            "One of the n values is empty. Please provide a "
                            "value.");
                    continue;
                }
                cfg.n_values.push_back(std::stoi(n));
            }

            // Set isPipeline to true if more than one n value is present
            isPipeline = (n_values.size() > 1) ? true : isPipeline;

            if (id < fields.size() && !(fields[id].empty())) {
                auto postOps = split(fields[id], ':');
                auto binary_post_op_pos = 0;
                for (auto i = 0; i < postOps.size(); i++) {
                    if (!postOps[i].empty()) {
                        cfg.post_ops.push_back(strToPostOps(postOps[i]));
                        // Track positions of binary post-operations
                        if (postOps[i].find("binary_") == 0) {
                            cfg.binary_post_ops_pos.push_back(
                                    binary_post_op_pos);
                        }
                        binary_post_op_pos++;
                    }
                }
            } else if (!options.post_ops.empty()) {
                // Short-format input rows omit the [postOp] column; fall back to the
                // global --post_ops chain so CLI defaults reach every row. Mirrors
                // inputCommandLineParser's handling of options.post_ops.
                auto binary_post_op_pos = 0;
                for (auto i = 0; i < options.post_ops.size(); i++) {
                    cfg.post_ops.push_back(options.post_ops[i]);
                    if (options.post_ops[i] == post_op_type_t::binary_add
                            || options.post_ops[i]
                                    == post_op_type_t::binary_mul) {
                        cfg.binary_post_ops_pos.push_back(binary_post_op_pos);
                    }
                    binary_post_op_pos++;
                }
            }
            id++;

            if (id < fields.size() && !(fields[id].empty())) {
                std::string bias_flag = fields[id];
                std::transform(bias_flag.begin(), bias_flag.end(),
                        bias_flag.begin(), ::tolower);
                if (bias_flag == "true" || bias_flag == "1") {
                    cfg.isBiasEnabled = true;
                } else if (bias_flag == "false" || bias_flag == "0") {
                    cfg.isBiasEnabled = false;
                } else {
                    commonlog_error(
                            "Invalid value for isBiasEnabled. Use true/false "
                            "or 1/0.");
                    continue;
                }
            } else {
                // Short-format input rows omit the [isBiasEnabled] column; fall back
                // to --bias (defaults to false in global_options).
                cfg.isBiasEnabled = options.isBiasEnabled;
            }

            cfg.iters = options.iters;
            cfg.dt.push_back(options.sdt);
            cfg.dt.push_back(options.wdt);
            cfg.dt.push_back(options.ddt);
            zendnnl::common::matmul_config_t &matmul_config
                    = zendnnl::common::matmul_config_t::instance();
            int32_t algo_ = options.ndims > 2 ? matmul_config.get_bmm_algo()
                                              : matmul_config.get_algo();
            matmul_algo_t algo = static_cast<matmul_algo_t>(algo_);
            if (algo == matmul_algo_t::none) {
                cfg.kernel_name = options.kernel_name;
                if (!validateMatmulKernelName(cfg.kernel_name)) {
                    auto kernel_name = options.ndims > 2 ? "aocl_dlp"
                                                         : "aocl_dlp_blocked";
                    commonlog_warning("Unknown kernel name '", cfg.kernel_name,
                            "'. Supported: aocl_dlp_blocked, onednn_blocked, "
                            "libxsmm_blocked, aocl_dlp, onednn, libxsmm, "
                            "batched_sgemm, auto, dynamic_dispatch, reference. "
                            "Using '",
                            kernel_name, "' instead.");
                    cfg.kernel_name = kernel_name;
                }
            } else {
                cfg.kernel_name = algoToStr(algo);
            }
            if (cfg.binary_post_ops_pos.size() > 0) {
                cfg.post_op_dt = options.post_op_dt;
            }
            cfg.is_weights_const = options.is_weights_const >= 0
                    ? options.is_weights_const
                    : (options.ndims > 2 ? 0 : 1);
            cfg.bias_dt = options.bias_dt;
            cfg.isTransA = options.isTransA;
            cfg.isTransB = options.isTransB;
            cfg.alpha = options.alpha;
            cfg.beta = options.beta;
            cfg.scale_granularity
                    = ((options.wdt == data_type_t::s4
                               || options.wdt == data_type_t::s8)
                              && options.scale_granularity == "none")
                    ? "channel"
                    : options.scale_granularity;
            cfg.group_size = options.group_size;
            cfg.scale_dt = options.scale_dt;
            cfg.warmup_iters = options.warmup_iters < 0 ? (cfg.iters) * 0.2
                                                        : options.warmup_iters;
            cfg.src_dynamic_quant = options.src_dynamic_quant;
            cfg.src_scale_granularity = options.src_scale_granularity;
            cfg.src_group_size = options.src_group_size;
            cfg.src_scale_dt = options.src_scale_dt;

            sync_src_weight_group_sizes(cfg);
            normalize_w4a8_quant_config(cfg);
            configs.push_back(cfg);
        } catch (const std::exception &e) {
            commonlog_error(e.what());
            continue;
        }
    }
}

void inputCommandLineParser(std::vector<MatmulConfig> &configs,
        bool &isPipeline, const global_options &options) {
    MatmulConfig cfg;
    try {
        if (options.ndims > 2) {
            cfg.bs = options.bs;
        } else {
            cfg.bs = 1;
        }
        cfg.m = options.m;
        cfg.k = options.k;
        if (options.n_values.size() > 1) { isPipeline = true; }
        for (const auto &n : options.n_values) {
            cfg.n_values.push_back(n);
        }
        cfg.isBiasEnabled = options.isBiasEnabled;
        cfg.iters = options.iters;
        cfg.dt.push_back(options.sdt);
        cfg.dt.push_back(options.wdt);
        cfg.dt.push_back(options.ddt);
        zendnnl::common::matmul_config_t &matmul_config
                = zendnnl::common::matmul_config_t::instance();
        int32_t algo_ = options.ndims > 2 ? matmul_config.get_bmm_algo()
                                          : matmul_config.get_algo();
        matmul_algo_t algo = static_cast<matmul_algo_t>(algo_);
        if (algo == matmul_algo_t::none) {
            cfg.kernel_name = options.kernel_name;
            if (!validateMatmulKernelName(cfg.kernel_name)) {
                auto kernel_name
                        = options.ndims > 2 ? "aocl_dlp" : "aocl_dlp_blocked";
                commonlog_warning("Unknown kernel name '", cfg.kernel_name,
                        "'. Supported: aocl_dlp_blocked, onednn_blocked, "
                        "libxsmm_blocked, aocl_dlp, onednn, libxsmm, "
                        "batched_sgemm, auto, dynamic_dispatch, reference. "
                        "Using '",
                        kernel_name, "' instead.");
                cfg.kernel_name = kernel_name;
            }
        } else {
            cfg.kernel_name = algoToStr(algo);
        }

        cfg.is_weights_const = options.is_weights_const >= 0
                ? options.is_weights_const
                : (options.ndims > 2 ? 0 : 1);
        cfg.bias_dt = options.bias_dt;
        if (options.post_ops.size() > 0) {
            auto binary_post_op_pos = 0;
            for (auto i = 0; i < options.post_ops.size(); i++) {
                cfg.post_ops.push_back(options.post_ops[i]);
                // Track positions of binary post-operations
                if (options.post_ops[i] == post_op_type_t::binary_add
                        || options.post_ops[i] == post_op_type_t::binary_mul) {
                    cfg.binary_post_ops_pos.push_back(binary_post_op_pos);
                }
                binary_post_op_pos++;
            }
        }
        if (cfg.binary_post_ops_pos.size() > 0) {
            cfg.post_op_dt = options.post_op_dt;
        }
        cfg.isTransA = options.isTransA;
        cfg.isTransB = options.isTransB;
        cfg.alpha = options.alpha;
        cfg.beta = options.beta;
        cfg.scale_granularity = ((options.wdt == data_type_t::s4
                                         || options.wdt == data_type_t::s8)
                                        && options.scale_granularity == "none")
                ? "channel"
                : options.scale_granularity;
        cfg.group_size = options.group_size;
        cfg.scale_dt = options.scale_dt;
        cfg.warmup_iters = options.warmup_iters < 0 ? (cfg.iters) * 0.2
                                                    : options.warmup_iters;
        cfg.src_dynamic_quant = options.src_dynamic_quant;
        cfg.src_scale_granularity = options.src_scale_granularity;
        cfg.src_group_size = options.src_group_size;
        cfg.src_scale_dt = options.src_scale_dt;

        sync_src_weight_group_sizes(cfg);
        normalize_w4a8_quant_config(cfg);
        configs.push_back(cfg);
    } catch (const std::exception &e) { commonlog_error(e.what()); }
}

namespace {

// Complete, self-contained description of one sweep dtype. There is no
// cross-dtype inheritance: every field a config needs is spelled out here so
// each dtype's parameter set stays independent and easy to audit. The
// per-config field requirements are not duplicated here; validate_dtype_fields()
// derives them from the dtype/quant fields, so the sweep reuses the exact same
// validation as the --input_file path.
struct SweepDTypeSpec {
    const char *name; // canonical name (matches a --dtype_sweep token)
    data_type_t sdt; // src (activation) dtype
    data_type_t wdt; // weight dtype
    data_type_t ddt; // dst dtype
    const char *kernel; // default kernel when algo is unset
    bool requires_lowoha; // dtype only valid on the LOWOHA path
    const char *wei_scale_granularity; // "none" | "channel" | "group"
    uint64_t wei_group_size; // weight group size (per-group only)
    data_type_t wei_scale_dt; // weight scale dtype
    bool src_dynamic_quant; // enable dynamic activation quant
    const char
            *src_scale_granularity; // "per-tensor" | "per-token" | "per-group"
    uint64_t src_group_size; // src group size (per-group only)
    data_type_t src_scale_dt; // src scale dtype
};

// The sweep catalog is not spelled out field-by-field. Instead each entry is
// described by a few orthogonal axes and the full SweepDTypeSpec is derived
// from them (see derive_sweep_spec), so adding a config is a one-line axis row
// and the low-level fields stay internally consistent by construction.
//
//   weight dtype  : f32 | bf16 | s8 | s4
//   quant scheme  : how the *activation* (src) is handled
//   weight gran   : per-channel (group 0) | per-group (group N)
//   src gran      : per-channel/per-token (group 0) | per-group (group N)
//
// Weight and src granularity are independent axes: the runtime classifies the
// src-scale granularity from the src-scale tensor shape, so e.g. per-group
// weights can pair with per-token activation scales. The derivation rules
// mirror validate_dtype_fields.
enum class QuantScheme {
    kNone, // plain float weights, no quantization
    kWoq, // weight-only quant (int4/int8 weights, bf16 activations, no src quant)
    kDynamic, // dynamic activation quant (INT8 dynamic / W4A8): bf16 src quantized at runtime
    kStatic, // static activation quant: integer (s8) src with a precomputed scale
};

enum class QuantGran {
    kPerChannel, // per-channel weights / per-token activations; group size 0
    kPerGroup, // per-group; group size N
};

// One orthogonal-axis description of a sweep config. `name` is the stable
// --dtype_sweep token; every other SweepDTypeSpec field is derived from these.
// Weight and src granularity are separate so decoupled combinations (e.g.
// per-group weights + per-token activations) are expressible.
struct SweepAxis {
    const char *name;
    data_type_t wdt; // weight dtype (f32 | bf16 | s8 | s4)
    data_type_t ddt; // dst dtype (src dtype is derived from the scheme)
    QuantScheme scheme;
    QuantGran wei_gran; // weight scale granularity (quantized schemes)
    uint64_t wei_group_size; // weight group size (per-group only)
    QuantGran src_gran; // src scale granularity (dynamic/static schemes)
    uint64_t src_group_size; // src group size (per-group only)
    data_type_t scale_dt; // weight+src scale dtype (quantized schemes only)
};

// Canonical sweep configs as orthogonal axes. Order defines the --dtype_sweep
// index and the "all" expansion; the first six preserve the historical indices.
// Layout: {name, wdt, ddt, scheme, wei_gran, wei_grp, src_gran, src_grp, scale_dt}.
static const SweepAxis kSweepAxes[] = {
        // 0-1: plain float, no quantization.
        {"bf16", data_type_t::bf16, data_type_t::bf16, QuantScheme::kNone,
                QuantGran::kPerChannel, 0, QuantGran::kPerChannel, 0,
                data_type_t::f32},
        {"fp32", data_type_t::f32, data_type_t::f32, QuantScheme::kNone,
                QuantGran::kPerChannel, 0, QuantGran::kPerChannel, 0,
                data_type_t::f32},
        // 2-3: int8 dynamic (s8 weights). Src and weight granularity are
        // independent: these two tokens keep the historical paired layouts
        // (both per-group, or per-token src + per-channel wei). Mixed
        // per-token src + per-group wei is accepted by validate_dtype_fields
        // on --input_file rows.
        {"int8_per_group", data_type_t::s8, data_type_t::bf16,
                QuantScheme::kDynamic, QuantGran::kPerGroup, 32,
                QuantGran::kPerGroup, 32, data_type_t::bf16},
        {"int8_per_token", data_type_t::s8, data_type_t::bf16,
                QuantScheme::kDynamic, QuantGran::kPerChannel, 0,
                QuantGran::kPerChannel, 0, data_type_t::bf16},
        // 4-5: int4 weight-only quant (WoQ). No src quant.
        {"int4_per_group", data_type_t::s4, data_type_t::bf16,
                QuantScheme::kWoq, QuantGran::kPerGroup, 32,
                QuantGran::kPerChannel, 0, data_type_t::bf16},
        {"int4_per_token", data_type_t::s4, data_type_t::bf16,
                QuantScheme::kWoq, QuantGran::kPerChannel, 0,
                QuantGran::kPerChannel, 0, data_type_t::bf16},
        // 6-7: int4 dynamic (W4A8). Per-group weights; src granularity varies.
        {"int4_dyn_per_group", data_type_t::s4, data_type_t::bf16,
                QuantScheme::kDynamic, QuantGran::kPerGroup, 32,
                QuantGran::kPerGroup, 32, data_type_t::bf16},
        {"int4_dyn_per_token", data_type_t::s4, data_type_t::bf16,
                QuantScheme::kDynamic, QuantGran::kPerGroup, 32,
                QuantGran::kPerChannel, 0, data_type_t::bf16},
        // 8-9: int8 static, integer (s8) dst. Integer output only supports a
        // per-channel weight scale + per-tensor static source scale, so the
        // validator forces both entries to that layout (the per_group token folds
        // onto the per_channel config and dedups under --dtype_sweep=all).
        {"int8_static_s8_per_group", data_type_t::s8, data_type_t::s8,
                QuantScheme::kStatic, QuantGran::kPerGroup, 32,
                QuantGran::kPerGroup, 32, data_type_t::bf16},
        {"int8_static_s8_per_token", data_type_t::s8, data_type_t::s8,
                QuantScheme::kStatic, QuantGran::kPerChannel, 0,
                QuantGran::kPerChannel, 0, data_type_t::bf16},
        // 10-11: int8 static, bf16 dst. Per-token/per-group static src scales are
        // supported with dequantized (bf16) output.
        {"int8_static_bf16_per_group", data_type_t::s8, data_type_t::bf16,
                QuantScheme::kStatic, QuantGran::kPerGroup, 32,
                QuantGran::kPerGroup, 32, data_type_t::bf16},
        {"int8_static_bf16_per_token", data_type_t::s8, data_type_t::bf16,
                QuantScheme::kStatic, QuantGran::kPerChannel, 0,
                QuantGran::kPerChannel, 0, data_type_t::bf16},
};

// Derive the full, low-level spec for one axis row. Keeps the derivation in one
// place so the catalog stays consistent with validate_dtype_fields.
static SweepDTypeSpec derive_sweep_spec(const SweepAxis &ax) {
    SweepDTypeSpec s {};
    s.name = ax.name;
    s.wdt = ax.wdt;
    s.ddt = ax.ddt;
    // Source (activation) dtype follows the scheme: float schemes keep the
    // weight's float dtype, static int8 keeps an integer source, and
    // weight-only / dynamic schemes feed a bf16 activation.
    switch (ax.scheme) {
        case QuantScheme::kNone: s.sdt = ax.wdt; break;
        case QuantScheme::kStatic: s.sdt = data_type_t::s8; break;
        default: s.sdt = data_type_t::bf16; break;
    }

    const bool quantized = ax.scheme != QuantScheme::kNone;
    s.kernel = quantized ? "aocl_dlp" : "aocl_dlp_blocked";
    s.requires_lowoha = quantized;

    const bool wei_per_group = ax.wei_gran == QuantGran::kPerGroup;
    if (ax.scheme == QuantScheme::kNone) {
        s.wei_scale_granularity = "none";
        s.wei_group_size = 0;
        s.wei_scale_dt = data_type_t::f32;
    } else {
        s.wei_scale_granularity = wei_per_group ? "group" : "channel";
        s.wei_group_size = wei_per_group ? ax.wei_group_size : 0;
        s.wei_scale_dt = ax.scale_dt;
    }

    const bool has_src_scale = ax.scheme == QuantScheme::kDynamic
            || ax.scheme == QuantScheme::kStatic;
    if (has_src_scale) {
        // Dynamic quantizes the source at runtime; static carries a precomputed
        // integer source with the same scale-tensor granularity. Either way the
        // src granularity is independent of the weights, and the kernel requires
        // the src scale dtype to equal the weight scale dtype.
        const bool src_per_group = ax.src_gran == QuantGran::kPerGroup;
        s.src_dynamic_quant = ax.scheme == QuantScheme::kDynamic;
        s.src_scale_granularity = src_per_group ? "per-group" : "per-token";
        s.src_group_size = src_per_group ? ax.src_group_size : 0;
        s.src_scale_dt = ax.scale_dt;
    } else {
        // none / woq: no source scale.
        s.src_dynamic_quant = false;
        s.src_scale_granularity = "per-tensor";
        s.src_group_size = 0;
        s.src_scale_dt = data_type_t::f32;
    }
    return s;
}

// Materialize the derived catalog once. Order matches kSweepAxes, so the
// --dtype_sweep index and "all" ordering are unchanged for existing entries.
static const std::vector<SweepDTypeSpec> &sweep_dtypes() {
    static const std::vector<SweepDTypeSpec> catalog = [] {
        std::vector<SweepDTypeSpec> v;
        v.reserve(sizeof(kSweepAxes) / sizeof(kSweepAxes[0]));
        for (const auto &ax : kSweepAxes) {
            v.push_back(derive_sweep_spec(ax));
        }
        return v;
    }();
    return catalog;
}

static size_t kNumSweepDTypes() {
    return sweep_dtypes().size();
}

// Comma-separated list of the canonical dtype names, built from the catalog so
// the accepted-values hint stays in sync automatically.
static std::string sweep_dtype_names_csv() {
    std::string s;
    const auto &catalog = sweep_dtypes();
    for (size_t i = 0; i < catalog.size(); ++i) {
        s += (i ? ", " : "");
        s += catalog[i].name;
    }
    return s;
}

static constexpr size_t kDefaultMSweep[] = {
        1,
        4,
        8,
        16,
        32,
        64,
        512,
        1024,
        2048,
};

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static const SweepDTypeSpec &lookup_dtype(size_t idx) {
    const auto &catalog = sweep_dtypes();
    if (idx >= catalog.size()) {
        throw std::invalid_argument("Unknown sweep dtype index");
    }
    return catalog[idx];
}

// name -> catalog index, derived once from the catalog itself.
static const std::unordered_map<std::string, size_t> &sweep_dtype_name_map() {
    static const std::unordered_map<std::string, size_t> names = [] {
        std::unordered_map<std::string, size_t> m;
        const auto &catalog = sweep_dtypes();
        for (size_t i = 0; i < catalog.size(); ++i) {
            m.emplace(catalog[i].name, i);
        }
        return m;
    }();
    return names;
}

static std::vector<size_t> parse_m_sweep_values(const std::string &s) {
    if (s.empty()) {
        return {std::begin(kDefaultMSweep), std::end(kDefaultMSweep)};
    }
    std::vector<size_t> out;
    for (const auto &token : split(s, ':')) {
        if (token.empty()) { continue; }
        const size_t m = std::stoull(token);
        if (m == 0) {
            throw std::invalid_argument("M sweep values must be > 0");
        }
        out.push_back(m);
    }
    if (out.empty()) {
        throw std::invalid_argument("No valid M values in --m_sweep");
    }
    return out;
}

static size_t parse_sweep_dtype_index(const std::string &token) {
    const auto &names = sweep_dtype_name_map();
    const auto it = names.find(to_lower(token));
    if (it == names.end()) {
        throw std::invalid_argument("Unknown sweep dtype '" + token
                + "'. Use all or: " + sweep_dtype_names_csv());
    }
    return it->second;
}

static std::vector<size_t> parse_sweep_dtype_indices(const std::string &s) {
    if (s.empty()) { return {}; }
    if (to_lower(s) == "all") {
        std::vector<size_t> all;
        all.reserve(kNumSweepDTypes());
        for (size_t i = 0; i < kNumSweepDTypes(); ++i) {
            all.push_back(i);
        }
        return all;
    }
    std::vector<size_t> out;
    for (const auto &token : split(s, ',')) {
        if (token.empty()) { continue; }
        const size_t idx = parse_sweep_dtype_index(token);
        if (std::find(out.begin(), out.end(), idx) == out.end()) {
            out.push_back(idx);
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("No valid dtypes in --dtype_sweep");
    }
    return out;
}

// Human-readable cache mode used in the expansion table, dedup signature and
// per-config results output.
static const char *cache_mode_to_str(CacheMode mode) {
    switch (mode) {
        case CacheMode::COLD: return "cold";
        case CacheMode::WARM: return "warm";
        case CacheMode::HOT: return "hot";
    }
    return "hot";
}

// Parse the --cache_sweep list (comma-separated hot/cold/warm). An empty string
// means "no cache sweep": the returned vector is empty and callers keep each
// config's inherited cache_mode (the global --cache_mode). Duplicates collapse.
static std::vector<CacheMode> parse_cache_sweep_modes(const std::string &s) {
    std::vector<CacheMode> out;
    if (s.empty()) { return out; }
    for (const auto &token : split(s, ',')) {
        if (token.empty()) { continue; }
        const std::string t = to_lower(token);
        CacheMode mode;
        if (t == "cold") {
            mode = CacheMode::COLD;
        } else if (t == "warm") {
            mode = CacheMode::WARM;
        } else if (t == "hot") {
            mode = CacheMode::HOT;
        } else {
            throw std::invalid_argument("Unknown cache mode '" + token
                    + "'. Use hot, cold or warm.");
        }
        if (std::find(out.begin(), out.end(), mode) == out.end()) {
            out.push_back(mode);
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("No valid cache modes in --cache_sweep");
    }
    return out;
}

static void resolve_kernel(MatmulConfig &cfg, const global_options &options,
        const char *dtype_kernel) {
    zendnnl::common::matmul_config_t &matmul_config
            = zendnnl::common::matmul_config_t::instance();
    const int32_t algo_ = options.ndims > 2 ? matmul_config.get_bmm_algo()
                                            : matmul_config.get_algo();
    const matmul_algo_t algo = static_cast<matmul_algo_t>(algo_);
    if (algo == matmul_algo_t::none) {
        cfg.kernel_name = dtype_kernel;
    } else {
        cfg.kernel_name = algoToStr(algo);
    }
}

// Merge a catalog dtype onto a base config. The input file has priority: only
// fields the row did not provide (see MatmulConfig::provided) are filled from
// the catalog, which acts purely as a default source. Numeric/bool quant
// fields follow their granularity flag, matching how the parser records them.
//
// Exception: when --dtype_sweep is explicitly present (force_profile), the
// swept dtype's full profile -- dt plus the weight/src quant granularity,
// group sizes and scale dtypes -- overrides the file so the dtype axis truly
// sweeps (e.g. a per-token entry becomes per-token even if the row was
// per-group). Non-quant fields (kernel, bias, transpose, alpha/beta, iters)
// still come from the file.
static void apply_sweep_dtype(MatmulConfig &cfg, const SweepDTypeSpec &spec,
        const global_options &options, bool force_profile) {
    if (force_profile || !cfg.provided.dt) {
        cfg.dt = {spec.sdt, spec.wdt, spec.ddt};
    }
    if (!cfg.provided.kernel) { resolve_kernel(cfg, options, spec.kernel); }
    if (force_profile || !cfg.provided.wei_scale) {
        cfg.scale_granularity = spec.wei_scale_granularity;
        cfg.group_size = spec.wei_group_size;
    }
    if (force_profile || !cfg.provided.wei_scale_dt) {
        cfg.scale_dt = spec.wei_scale_dt;
    }
    if (force_profile || !cfg.provided.src_scale) {
        cfg.src_dynamic_quant = spec.src_dynamic_quant;
        cfg.src_scale_granularity = spec.src_scale_granularity;
        cfg.src_group_size = spec.src_group_size;
    }
    if (force_profile || !cfg.provided.src_scale_dt) {
        cfg.src_scale_dt = spec.src_scale_dt;
    }
}

// --- M sweep --------------------------------------------------------------
// Expand one base config across the M sweep values. M is the only field that
// varies here; everything else is inherited from the base row.
static std::vector<MatmulConfig> expand_m_sweep(
        const MatmulConfig &base, const std::vector<size_t> &m_values) {
    std::vector<MatmulConfig> out;
    out.reserve(m_values.size());
    for (const size_t m : m_values) {
        MatmulConfig cfg = base;
        cfg.m = m;
        out.push_back(std::move(cfg));
    }
    return out;
}

// --- dtype sweep ----------------------------------------------------------
// Expand one config across the requested catalog dtypes. Each entry applies its
// dtype/quant profile (subject to force_profile), is checked with the shared
// --input_file validator, and normalized. Entries that are invalid or require
// the LOWOHA path when it is off are dropped.
static std::vector<MatmulConfig> expand_dtype_sweep(const MatmulConfig &base,
        const std::vector<size_t> &dtype_indices, const global_options &options,
        bool force_profile, bool is_lowoha) {
    std::vector<MatmulConfig> out;
    if (dtype_indices.empty()) {
        out.push_back(base);
        return out;
    }
    out.reserve(dtype_indices.size());
    for (const size_t dtype_idx : dtype_indices) {
        const SweepDTypeSpec &spec = lookup_dtype(dtype_idx);
        if (spec.requires_lowoha && !is_lowoha) { continue; }
        MatmulConfig cfg = base;
        apply_sweep_dtype(cfg, spec, options, force_profile);
        // Reuse the exact same dtype/quant validation as the --input_file path
        // instead of a parallel sweep-only validator.
        const std::string ctx = std::string("sweep dtype '") + spec.name
                + "' (k=" + std::to_string(cfg.k)
                + ", n=" + std::to_string(cfg.n_values[0]) + ")";
        if (!validate_dtype_fields(cfg, ctx)) { continue; }
        normalize_w4a8_quant_config(cfg);
        out.push_back(std::move(cfg));
    }
    return out;
}

// --- cache sweep ----------------------------------------------------------
// Expand one config across the requested cache modes. cache_mode is a
// process-independent, per-config measurement setting, so each mode simply
// becomes its own config; the benchmark driver reads cfg.cache_mode. When the
// mode list is empty (no --cache_sweep) the config passes through unchanged,
// keeping its inherited global --cache_mode.
static std::vector<MatmulConfig> expand_cache_sweep(
        const MatmulConfig &base, const std::vector<CacheMode> &cache_modes) {
    std::vector<MatmulConfig> out;
    if (cache_modes.empty()) {
        out.push_back(base);
        return out;
    }
    out.reserve(cache_modes.size());
    for (const CacheMode mode : cache_modes) {
        MatmulConfig cfg = base;
        cfg.cache_mode = mode;
        out.push_back(std::move(cfg));
    }
    return out;
}

} // namespace

// Display helpers (defined further below) reused for the sweep expansion table.
static std::string disp_wei_group_size(const MatmulConfig &c);
static std::string disp_wei_scale_dt(const MatmulConfig &c);
static std::string disp_src_scale_granularity(const MatmulConfig &c);
static std::string disp_src_group_size(const MatmulConfig &c);
static std::string disp_src_scale_dt(const MatmulConfig &c);

// Print the fully expanded sweep as an aligned table so the exact set of
// generated configurations is visible before benchmarking begins.
static void print_sweep_expansion_table(
        const std::vector<MatmulConfig> &out, const global_options &options) {
    const bool has_bs = options.ndims > 2;
    struct Col {
        const char *name;
        int width;
    };
    std::vector<Col> cols = {{"#", 4}};
    if (has_bs) { cols.push_back({"BS", 6}); }
    const std::vector<Col> tail = {{"M", 7}, {"K", 8}, {"N", 8}, {"dt", 16},
            {"kernel", 18}, {"w_gran", 9}, {"w_grp", 7}, {"w_sdt", 7},
            {"src_dq", 7}, {"src_gran", 11}, {"src_grp", 8}, {"src_sdt", 8},
            {"cache", 6}};
    cols.insert(cols.end(), tail.begin(), tail.end());

    auto print_row = [&](const std::vector<std::string> &vals) {
        for (size_t j = 0; j < cols.size(); ++j) {
            std::cout << std::left << std::setw(cols[j].width)
                      << (j < vals.size() ? vals[j] : std::string()) << ' ';
        }
        std::cout << '\n';
    };

    std::vector<std::string> header;
    size_t rule_width = 0;
    for (const auto &c : cols) {
        header.emplace_back(c.name);
        rule_width += c.width + 1;
    }
    std::cout << "Sweep expansion table (" << out.size() << " config(s)):\n";
    print_row(header);
    std::cout << std::string(rule_width, '-') << '\n';

    for (size_t i = 0; i < out.size(); ++i) {
        const MatmulConfig &c = out[i];
        std::vector<std::string> vals;
        vals.push_back(std::to_string(i));
        if (has_bs) { vals.push_back(std::to_string(c.bs)); }
        vals.push_back(std::to_string(c.m));
        vals.push_back(std::to_string(c.k));
        vals.push_back(std::to_string(c.n_values[0]));
        vals.push_back(datatypeToStr(c.dt[0]) + ":" + datatypeToStr(c.dt[1])
                + ":" + datatypeToStr(c.dt[2]));
        vals.push_back(c.kernel_name);
        vals.push_back(c.scale_granularity);
        vals.push_back(disp_wei_group_size(c));
        vals.push_back(disp_wei_scale_dt(c));
        vals.push_back(std::to_string(c.src_dynamic_quant));
        vals.push_back(disp_src_scale_granularity(c));
        vals.push_back(disp_src_group_size(c));
        vals.push_back(disp_src_scale_dt(c));
        vals.push_back(cache_mode_to_str(c.cache_mode));
        print_row(vals);
    }
    std::cout << std::flush;
}

std::vector<MatmulConfig> expand_matmul_sweep(
        const std::vector<MatmulConfig> &base, const global_options &options,
        bool is_lowoha) {
    const auto m_values = parse_m_sweep_values(options.m_sweep_str);
    const auto dtype_indices
            = parse_sweep_dtype_indices(options.dtype_sweep_str);
    const auto cache_modes = parse_cache_sweep_modes(options.cache_sweep_str);

    // An explicit --dtype_sweep string means the user wants to sweep the dtype
    // itself: the swept dtype's full quant profile (dt + granularity + group
    // sizes + scale dtypes) overrides the file, so per-token vs per-group etc.
    // actually vary. Non-quant fields still come from the file. With no
    // --dtype_sweep it stays an M-only sweep and the file's dtype is preserved.
    const bool force_profile = !options.dtype_sweep_str.empty();

    std::vector<MatmulConfig> out;
    const size_t cache_factor = cache_modes.empty() ? 1 : cache_modes.size();
    const size_t dtype_factor
            = dtype_indices.empty() ? 1 : dtype_indices.size();
    out.reserve(base.size() * m_values.size() * dtype_factor * cache_factor);

    // A benchmark is uniquely identified by the shape plus the full dtype/quant
    // config it ends up running, so dedup on the final config signature rather
    // than on shape alone. This keeps distinct rows that share a shape but differ
    // in quant (e.g. per-group vs per-token) while still collapsing genuinely
    // identical configs -- whether they come from repeated input rows or from a
    // forced-dtype sweep where several catalog dtypes reduce to the same dt.
    std::set<std::string> seen_configs;
    size_t duplicate_configs = 0;
    auto config_signature = [](const MatmulConfig &c) {
        std::ostringstream os;
        os << c.bs << '|' << c.m << '|' << c.k << '|' << c.n_values[0] << '|';
        for (const auto d : c.dt) {
            os << static_cast<int>(d) << ',';
        }
        os << '|' << c.kernel_name << '|' << c.scale_granularity << '|'
           << c.group_size << '|' << static_cast<int>(c.scale_dt) << '|'
           << c.src_dynamic_quant << '|' << c.src_scale_granularity << '|'
           << c.src_group_size << '|' << static_cast<int>(c.src_scale_dt) << '|'
           << static_cast<int>(c.cache_mode);
        return os.str();
    };

    for (const auto &base_cfg : base) {
        if (base_cfg.n_values.size() != 1) {
            commonlog_warning("Skipping pipeline row '", base_cfg.modelName,
                    "' during sweep expansion.");
            continue;
        }
        if (base_cfg.k == 0 || base_cfg.n_values[0] == 0) { continue; }

        // Compose the independent axes: first vary M, then dtype on each
        // M-expanded config, then cache mode on each dtype-expanded config. Dedup
        // the fully-built configs.
        for (const MatmulConfig &m_cfg : expand_m_sweep(base_cfg, m_values)) {
            for (MatmulConfig &d_cfg : expand_dtype_sweep(m_cfg, dtype_indices,
                         options, force_profile, is_lowoha)) {
                for (MatmulConfig &cfg :
                        expand_cache_sweep(d_cfg, cache_modes)) {
                    if (!seen_configs.insert(config_signature(cfg)).second) {
                        duplicate_configs++;
                        continue;
                    }
                    out.push_back(std::move(cfg));
                }
            }
        }
    }

    if (out.empty()) {
        commonlog_warning("Sweep expansion produced no configurations.");
        return out;
    }

    std::cout << "Sweep expansion: " << base.size() << " input row(s) -> "
              << out.size() << " config(s) (M x dtype"
              << (cache_modes.empty() ? "" : " x cache") << " cross-product).";
    if (duplicate_configs > 0) {
        std::cout << " Skipped " << duplicate_configs
                  << " duplicate config(s).";
    }
    std::cout << std::endl;

    print_sweep_expansion_table(out, options);
    return out;
}

// Display helpers for the results tables/CSVs. Several quant columns hold
// non-empty defaults (e.g. src_scale_granularity="per-tensor", scale_dt="f32")
// even when quantization is inactive, which is misleading in output (a plain
// bf16 row appears to use per-tensor src quant). These blank out the columns
// that do not apply so an inactive default is not mistaken for a real setting.
static std::string disp_wei_group_size(const MatmulConfig &c) {
    return c.scale_granularity == "none" ? std::string()
                                         : std::to_string(c.group_size);
}
static std::string disp_wei_scale_dt(const MatmulConfig &c) {
    return c.scale_granularity == "none" ? std::string()
                                         : datatypeToStr(c.scale_dt);
}
// A config carries a source scale when it either dynamically quantizes the
// source (INT8 dynamic / W4A8) or feeds a statically-quantized integer source (s8/u8 src
// with s8 weights). Both cases have meaningful src-scale fields to display.
static bool config_has_src_scale(const MatmulConfig &c) {
    if (c.src_dynamic_quant) { return true; }
    return c.dt.size() >= 2
            && (c.dt[0] == data_type_t::s8 || c.dt[0] == data_type_t::u8)
            && c.dt[1] == data_type_t::s8;
}
static std::string disp_src_scale_granularity(const MatmulConfig &c) {
    return config_has_src_scale(c) ? c.src_scale_granularity : std::string();
}
static std::string disp_src_group_size(const MatmulConfig &c) {
    return config_has_src_scale(c) ? std::to_string(c.src_group_size)
                                   : std::string();
}
static std::string disp_src_scale_dt(const MatmulConfig &c) {
    return config_has_src_scale(c) ? datatypeToStr(c.src_scale_dt)
                                   : std::string();
}

void log_benchmark_failure(const MatmulConfig &cfg) {
    std::string post_op = "";
    if (!cfg.post_ops.empty()) {
        for (auto j = 0; j < cfg.post_ops.size(); j++) {
            post_op += (j > 0 ? ":" : "") + postOpsToStr(cfg.post_ops[j]);
        }
    }
    std::string n_values = "";
    for (auto i = 0; i < cfg.n_values.size(); i++) {
        n_values += (i > 0 ? ":" : "") + std::to_string(cfg.n_values[i]);
    }
    commonlog_error("Benchmark failed for ", cfg.m, ", ", cfg.k, ", ", n_values,
            ", ", datatypeToStr(cfg.dt[0]), ":", datatypeToStr(cfg.dt[1]), ":",
            datatypeToStr(cfg.dt[2]), ", ", cfg.isBiasEnabled, ", ",
            (cfg.isBiasEnabled ? datatypeToStr(cfg.bias_dt) : ""), ", ",
            post_op, ", ",
            (cfg.binary_post_ops_pos.size() > 0 ? datatypeToStr(cfg.post_op_dt)
                                                : ""),
            ", ", cfg.kernel_name, ", ", cfg.is_weights_const, ", ",
            cfg.isTransA, ", ", cfg.isTransB, ", ", cfg.alpha, ", ", cfg.beta,
            ", ", cfg.scale_granularity, ", ", cfg.group_size, ", ",
            datatypeToStr(cfg.scale_dt), ", ", cfg.warmup_iters, ", ",
            cfg.src_dynamic_quant, ", ", cfg.src_scale_granularity, ", ",
            cfg.src_group_size, ", ", datatypeToStr(cfg.src_scale_dt));
}

void print_matmul_execution_summary(const MatmulConfig &cfg,
        const std::vector<TimingStats> &time_stats_layer,
        const global_options &options) {
    std::string post_op = "";
    if (!cfg.post_ops.empty()) {
        for (auto j = 0; j < cfg.post_ops.size(); j++) {
            post_op += (j > 0 ? ":" : "") + postOpsToStr(cfg.post_ops[j]);
        }
    }
    std::string n_values = "";
    double total_time = 0.0;
    for (auto i = 0; i < cfg.n_values.size(); i++) {
        n_values += (i > 0 ? ":" : "") + std::to_string(cfg.n_values[i]);
        total_time += time_stats_layer[i].total_time_ms;
    }
    if (options.ndims > 2) { std::cout << cfg.bs << ", "; }
    std::cout << cfg.m << ", " << cfg.k << ", " << n_values << ", " << cfg.iters
              << ", " << datatypeToStr(cfg.dt[0]) << ":"
              << datatypeToStr(cfg.dt[1]) << ":" << datatypeToStr(cfg.dt[2])
              << ", " << cfg.isBiasEnabled << ", "
              << (cfg.isBiasEnabled ? datatypeToStr(cfg.bias_dt) : "") << ", "
              << post_op << ", "
              << (cfg.binary_post_ops_pos.size() > 0
                                 ? datatypeToStr(cfg.post_op_dt)
                                 : "")
              << ", " << cfg.kernel_name << ", " << cfg.is_weights_const << ", "
              << cfg.isTransA << ", " << cfg.isTransB << ", " << cfg.alpha
              << ", " << cfg.beta << ", " << cfg.scale_granularity << ", "
              << disp_wei_group_size(cfg) << ", " << disp_wei_scale_dt(cfg)
              << ", " << cfg.warmup_iters << ", " << cfg.src_dynamic_quant
              << ", " << disp_src_scale_granularity(cfg) << ", "
              << disp_src_group_size(cfg) << ", " << disp_src_scale_dt(cfg)
              << ", " << total_time << std::endl;
}

void write_each_config_result(const MatmulConfig &config,
        const std::vector<TimingStats> &stat, std::ostream &outfile,
        const bool isLOWOHA, int layer_num, double percentage,
        bool isPipeline) {

    size_t m = config.m;
    size_t k = (layer_num == 0) ? config.k : config.n_values[layer_num - 1];
    size_t n = config.n_values[layer_num];
    size_t bs = config.bs;
    double gops = (2 * bs * m * k * n * 0.000000001);
    double gflops_val
            = (gops / (stat[layer_num].total_time_ms / config.iters)) * 1000;
    outfile << m << ", " << k << ", " << n;
    outfile << ", " << config.iters << ", " << datatypeToStr(config.dt[0])
            << ":" << datatypeToStr(config.dt[1]) << ":"
            << datatypeToStr(config.dt[2]) << ", " << config.isBiasEnabled
            << ", "
            << (config.isBiasEnabled ? datatypeToStr(config.bias_dt) : "")
            << ", ";
    if (!config.post_ops.empty()) {
        outfile << postOpsToStr(config.post_ops[0]);
        for (size_t j = 1; j < config.post_ops.size(); ++j) {
            outfile << ":" << postOpsToStr(config.post_ops[j]);
        }
    }
    outfile << ", ";
    if (config.binary_post_ops_pos.size() > 0) {
        outfile << datatypeToStr(config.post_op_dt);
    }
    outfile << ", ";
    outfile << config.kernel_name << ", " << config.is_weights_const << ", "
            << config.isTransA << ", " << config.isTransB << ", "
            << config.alpha << ", " << config.beta << ", "
            << config.scale_granularity << ", " << disp_wei_group_size(config)
            << ", " << disp_wei_scale_dt(config) << ", " << config.warmup_iters
            << ", " << config.src_dynamic_quant << ", "
            << disp_src_scale_granularity(config) << ", "
            << disp_src_group_size(config) << ", " << disp_src_scale_dt(config)
            << ", " << stat[layer_num].total_time_ms << ", "
            << (stat[layer_num].total_time_ms / config.iters) << ", "
            << gflops_val;
    if (isPipeline) { outfile << ", " << percentage; }
#if MEASURE_INDIVIDUAL_TIMINGS
    if (!isLOWOHA) {
        double ctx_creation_percentage
                = (stat[layer_num].context_creation_ms
                          / stat[layer_num].total_time_ms)
                * 100;
        double op_creation_percentage = (stat[layer_num].operator_creation_ms
                                                / stat[layer_num].total_time_ms)
                * 100;
        double op_execution_percentage
                = (stat[layer_num].operator_execution_ms
                          / stat[layer_num].total_time_ms)
                * 100;
        outfile << ", " << stat[layer_num].context_creation_ms << " ("
                << ctx_creation_percentage << " %), "
                << stat[layer_num].operator_creation_ms << " ("
                << op_creation_percentage << " %), "
                << stat[layer_num].operator_execution_ms << " ("
                << op_execution_percentage << " %)";
    }
#endif
    outfile << std::endl;
}

void cal_column_width(const MatmulConfig &config,
        const std::vector<TimingStats> &stat, std::vector<size_t> &col_widths,
        int st_index, const bool isLOWOHA, int layer_num, double percentage,
        bool isPipeline) {
    size_t m = config.m;
    size_t k = ((layer_num == 0) ? config.k : config.n_values[layer_num - 1]);
    size_t n = config.n_values[layer_num];
    size_t bs = config.bs;
    double gops = (2 * bs * m * k * n * 0.000000001);
    double gflops_val
            = (gops / (stat[layer_num].total_time_ms / config.iters)) * 1000;
    int col = st_index;
    col_widths[col++] = std::max(col_widths[col], std::to_string(m).size() + 2);
    col_widths[col++] = std::max(col_widths[col], std::to_string(k).size() + 2);
    col_widths[col++] = std::max(col_widths[col], std::to_string(n).size() + 2);
    col_widths[col++] = std::max(
            col_widths[col], std::to_string(config.iters).size() + 2);
    std::string dt_str = datatypeToStr(config.dt[0]) + ":"
            + datatypeToStr(config.dt[1]) + ":" + datatypeToStr(config.dt[2]);
    col_widths[col++] = std::max(col_widths[col], dt_str.size() + 2);
    col_widths[col++] = std::max(
            col_widths[col], std::to_string(config.isBiasEnabled).size() + 2);
    std::string bias_dt_str
            = config.isBiasEnabled ? datatypeToStr(config.bias_dt) : "";
    col_widths[col++] = std::max(col_widths[col], bias_dt_str.size() + 2);
    std::string postop_str;
    if (!config.post_ops.empty()) {
        postop_str += postOpsToStr(config.post_ops[0]);
        for (size_t j = 1; j < config.post_ops.size(); ++j) {
            postop_str += ":" + postOpsToStr(config.post_ops[j]);
        }
    }
    col_widths[col++] = std::max(col_widths[col], postop_str.size() + 2);
    col_widths[col++] = std::max(col_widths[col],
            config.binary_post_ops_pos.size() > 0
                    ? datatypeToStr(config.post_op_dt).size() + 2
                    : 0);
    col_widths[col++]
            = std::max(col_widths[col], config.kernel_name.size() + 2);
    col_widths[col++] = std::max(col_widths[col],
            std::to_string(config.is_weights_const).size() + 2);
    col_widths[col++] = std::max(
            col_widths[col], std::to_string(config.isTransA).size() + 2);
    col_widths[col++] = std::max(
            col_widths[col], std::to_string(config.isTransB).size() + 2);
    col_widths[col++] = std::max(
            col_widths[col], std::to_string(config.alpha).size() + 2);
    col_widths[col++]
            = std::max(col_widths[col], std::to_string(config.beta).size() + 2);
    col_widths[col++]
            = std::max(col_widths[col], config.scale_granularity.size() + 2);
    col_widths[col++]
            = std::max(col_widths[col], disp_wei_group_size(config).size() + 2);
    col_widths[col++]
            = std::max(col_widths[col], disp_wei_scale_dt(config).size() + 2);
    col_widths[col++] = std::max(
            col_widths[col], std::to_string(config.warmup_iters).size() + 2);
    col_widths[col++] = std::max(col_widths[col],
            std::to_string(config.src_dynamic_quant).size() + 2);
    col_widths[col++] = std::max(
            col_widths[col], disp_src_scale_granularity(config).size() + 2);
    col_widths[col++]
            = std::max(col_widths[col], disp_src_group_size(config).size() + 2);
    col_widths[col++]
            = std::max(col_widths[col], disp_src_scale_dt(config).size() + 2);
    col_widths[col++] = std::max(col_widths[col],
            std::to_string((int)stat[0].total_time_ms).size() + 2);
    col_widths[col++] = std::max(col_widths[col],
            std::to_string((int)(stat[0].total_time_ms / config.iters)).size()
                    + 2);
    std::ostringstream gflops_ss;
    gflops_ss << std::fixed << std::setprecision(2) << gflops_val;
    col_widths[col++] = std::max(col_widths[col], gflops_ss.str().size() + 2);
    if (isPipeline) {
        std::ostringstream perc_ss;
        perc_ss << std::fixed << std::setprecision(2) << percentage << " %";
        col_widths[col++] = std::max(col_widths[col], perc_ss.str().size() + 2);
    }
#if MEASURE_INDIVIDUAL_TIMINGS
    if (!isLOWOHA) {
        std::ostringstream ctx_str, op_create_str, op_exec_str;
        double ctx_creation_percentage
                = (stat[0].context_creation_ms / stat[0].total_time_ms) * 100;
        double op_creation_percentage
                = (stat[0].operator_creation_ms / stat[0].total_time_ms) * 100;
        double op_execution_percentage
                = (stat[0].operator_execution_ms / stat[0].total_time_ms) * 100;
        ctx_str << std::fixed << std::setprecision(2)
                << stat[0].context_creation_ms << " ("
                << ctx_creation_percentage << " %)";
        op_create_str << std::fixed << std::setprecision(2)
                      << stat[0].operator_creation_ms << " ("
                      << op_creation_percentage << " %)";
        op_exec_str << std::fixed << std::setprecision(2)
                    << stat[0].operator_execution_ms << " ("
                    << op_execution_percentage << " %)";
        col_widths[col++] = std::max(col_widths[col], ctx_str.str().size() + 2);
        col_widths[col++]
                = std::max(col_widths[col], op_create_str.str().size() + 2);
        col_widths[col++]
                = std::max(col_widths[col], op_exec_str.str().size() + 2);
    }
#endif
}

void fill_row(const MatmulConfig &config, const std::vector<TimingStats> &stat,
        std::vector<std::string> &row, const bool isLOWOHA, int layer_num,
        double percentage, bool isPipeline) {
    size_t m = config.m;
    size_t k = ((layer_num == 0) ? config.k : config.n_values[layer_num - 1]);
    size_t n = config.n_values[layer_num];
    size_t bs = config.bs;
    double gops = (2 * bs * m * k * n * 0.000000001);
    double gflops_val
            = ((gops / (stat[layer_num].total_time_ms / config.iters)) * 1000);
    row.push_back(std::to_string(m));
    row.push_back(std::to_string(k));
    row.push_back(std::to_string(n));
    row.push_back(std::to_string(config.iters));
    row.push_back(datatypeToStr(config.dt[0]) + ":"
            + datatypeToStr(config.dt[1]) + ":" + datatypeToStr(config.dt[2]));
    row.push_back(std::to_string(config.isBiasEnabled));
    row.push_back(config.isBiasEnabled ? datatypeToStr(config.bias_dt) : "");
    std::string postop_str;
    if (!config.post_ops.empty()) {
        postop_str += postOpsToStr(config.post_ops[0]);
        for (size_t j = 1; j < config.post_ops.size(); ++j) {
            postop_str += ":" + postOpsToStr(config.post_ops[j]);
        }
    }
    row.push_back(postop_str);
    row.push_back(config.binary_post_ops_pos.size() > 0
                    ? datatypeToStr(config.post_op_dt)
                    : "");
    row.push_back(config.kernel_name);
    row.push_back(std::to_string(config.is_weights_const));
    row.push_back(std::to_string(config.isTransA));
    row.push_back(std::to_string(config.isTransB));
    row.push_back(std::to_string(config.alpha));
    row.push_back(std::to_string(config.beta));
    row.push_back(config.scale_granularity);
    row.push_back(disp_wei_group_size(config));
    row.push_back(disp_wei_scale_dt(config));
    row.push_back(std::to_string(config.warmup_iters));
    row.push_back(std::to_string(config.src_dynamic_quant));
    row.push_back(disp_src_scale_granularity(config));
    row.push_back(disp_src_group_size(config));
    row.push_back(disp_src_scale_dt(config));
    std::ostringstream total_time_ss;
    total_time_ss << std::fixed << std::setprecision(2)
                  << stat[layer_num].total_time_ms;
    row.push_back(total_time_ss.str());
    std::ostringstream avg_time_ss;
    avg_time_ss << std::fixed << std::setprecision(6)
                << (stat[layer_num].total_time_ms / config.iters);
    row.push_back(avg_time_ss.str());
    std::ostringstream gflops_ss;
    gflops_ss << std::fixed << std::setprecision(2) << gflops_val;
    row.push_back(gflops_ss.str());
    if (isPipeline) {
        std::ostringstream perc_ss;
        perc_ss << std::fixed << std::setprecision(2) << percentage << " %";
        row.push_back(perc_ss.str());
    }
#if MEASURE_INDIVIDUAL_TIMINGS
    if (!isLOWOHA) {
        std::ostringstream ctx_str, op_create_str, op_exec_str;
        double ctx_creation_percentage
                = (stat[layer_num].context_creation_ms
                          / stat[layer_num].total_time_ms)
                * 100;
        double op_creation_percentage = (stat[layer_num].operator_creation_ms
                                                / stat[layer_num].total_time_ms)
                * 100;
        double op_execution_percentage
                = (stat[layer_num].operator_execution_ms
                          / stat[layer_num].total_time_ms)
                * 100;
        ctx_str << std::fixed << std::setprecision(2)
                << stat[layer_num].context_creation_ms << " ("
                << ctx_creation_percentage << " %)";
        op_create_str << std::fixed << std::setprecision(2)
                      << stat[layer_num].operator_creation_ms << " ("
                      << op_creation_percentage << " %)";
        op_exec_str << std::fixed << std::setprecision(2)
                    << stat[layer_num].operator_execution_ms << " ("
                    << op_execution_percentage << " %)";
        row.push_back(ctx_str.str());
        row.push_back(op_create_str.str());
        row.push_back(op_exec_str.str());
    }
#endif
}

void log_pipeline_results(
        std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>>
                &matmul_results,
        std::ostream &outfile, const global_options &options,
        const InputMode inputMode) {

    outfile << std::fixed << std::setprecision(2);
    outfile << "Layer Number, ";
    if (inputMode == InputMode::MODEL) { outfile << "Model Name, "; }
    if (options.ndims > 2) { outfile << "BS, "; }
    outfile << "M, K, N, Iterations, Data type, Bias Enabled, Bias Data type, "
            << "Post Operation, PostOp Data type, "
            << "Kernel name, isWeightsConst, isTransA, isTransB, "
            << "Alpha, Beta, Weight Scale Granularity, Weight Group Size, "
               "Weight Scale Data type, Warmup iterations, "
            << "Src Dynamic Quant, Src Scale Granularity, Src Group Size, Src "
               "Scale Data type, "
            << "Total time (ms) (all iters), Avg time (ms), GFLOPS, % of Total";
#if MEASURE_INDIVIDUAL_TIMINGS
    outfile << ", Context Creation (ms & %), Operator Creation (ms & %), "
               "Operator Execution (ms & %)";
#endif
    outfile << std::endl;

    // Write results to CSV for each configuration
    for (const auto &result : matmul_results) {
        const MatmulConfig &config = result.first;
        const std::vector<TimingStats> &stat = result.second;
        double total_time = 0.0;
        for (auto i = 0; i < config.n_values.size(); i++) {
            total_time += stat[i].total_time_ms;
        }
        outfile << "Summary, ";
        if (inputMode == InputMode::MODEL) {
            outfile << config.modelName << ", ";
        }
        if (options.ndims > 2) { outfile << config.bs << ", "; }
        outfile << config.m << ", " << config.k << ", ";
        // Output N values separated by ':'
        if (!config.n_values.empty()) {
            outfile << config.n_values[0];
            for (size_t i = 1; i < config.n_values.size(); ++i) {
                outfile << ":" << config.n_values[i];
            }
        }
        outfile << ", " << config.iters << ", " << datatypeToStr(config.dt[0])
                << ":" << datatypeToStr(config.dt[1]) << ":"
                << datatypeToStr(config.dt[2]) << ", " << config.isBiasEnabled
                << ", "
                << (config.isBiasEnabled ? datatypeToStr(config.bias_dt) : "")
                << ", ";
        if (!config.post_ops.empty()) {
            outfile << postOpsToStr(config.post_ops[0]);
            for (size_t j = 1; j < config.post_ops.size(); ++j) {
                outfile << ":" << postOpsToStr(config.post_ops[j]);
            }
        }
        outfile << ", ";
        if (config.binary_post_ops_pos.size() > 0) {
            outfile << datatypeToStr(config.post_op_dt);
        }
        outfile << ", ";
        outfile << config.kernel_name << ", " << config.is_weights_const << ", "
                << config.isTransA << ", " << config.isTransB << ", "
                << config.alpha << ", " << config.beta << ", "
                << config.scale_granularity << ", "
                << disp_wei_group_size(config) << ", "
                << disp_wei_scale_dt(config) << ", " << config.warmup_iters
                << ", " << config.src_dynamic_quant << ", "
                << disp_src_scale_granularity(config) << ", "
                << disp_src_group_size(config) << ", "
                << disp_src_scale_dt(config) << ", " << total_time;
        outfile << std::endl;

        for (auto i = 0; i < stat.size(); i++) {
            double percentage = (stat[i].total_time_ms / total_time) * 100;
            outfile << "Layer " << i << ", ";
            if (inputMode == InputMode::MODEL) {
                outfile << config.modelName << ", ";
            }
            if (options.ndims > 2) { outfile << config.bs << ", "; }
            write_each_config_result(
                    config, stat, outfile, false, i, percentage, true);
        }
    }
}

void print_pipeline_results(
        std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>>
                &matmul_results,
        std::ostream &outfile, const global_options &options,
        const InputMode inputMode) {

    // Dynamic column widths calculation
    std::vector<std::string> headers = {"Layer"};
    if (inputMode == InputMode::MODEL) { headers.push_back("Model Name"); }
    if (options.ndims > 2) { headers.push_back("BS"); }
    headers.insert(headers.end(),
            {"M", "K", "N", "Iters", "Data_type", "Bias_Enabled", "Bias_dt",
                    "PostOp", "PostOp_dt", "Kernel_Name", "isWeightsConst",
                    "isTransA", "isTransB", "Alpha", "Beta",
                    "Weight_Scale_Granularity", "Weight_Group_Size",
                    "Weight_Scale_dt", "Warmup_iters", "Src_Dynamic_Quant",
                    "Src_Scale_Granularity", "Src_Group_Size", "Src_Scale_dt",
                    "Total_time(ms, all iters)", "Avg_time(ms)", "GFLOPS",
                    "%_of_Total"});
#if MEASURE_INDIVIDUAL_TIMINGS
    headers.push_back("Ctx_Creation(ms_%)");
    headers.push_back("Op_Creation(ms_%)");
    headers.push_back("Op_Execution(ms_%)");
#endif
    std::vector<size_t> col_widths(headers.size());
    // Initialize with header lengths
    for (size_t i = 0; i < headers.size(); ++i) {
        col_widths[i] = headers[i].size() + 2;
    }
    // Compute max width for each column based on all data rows
    for (const auto &result : matmul_results) {
        const MatmulConfig &config = result.first;
        const std::vector<TimingStats> &stat = result.second;
        double total_time = 0.0;
        for (auto i = 0; i < config.n_values.size(); i++) {
            total_time += stat[i].total_time_ms;
        }
        // Update column widths for summary and per-layer rows
        int col = 0;
        col_widths[col++]
                = std::max(col_widths[col], std::string("Summary").size() + 2);
        if (inputMode == InputMode::MODEL) {
            col_widths[col++] = std::max(
                    col_widths[col], std::string("Model Name").size() + 2);
        }
        if (options.ndims > 2) {
            col_widths[col++]
                    = std::max(col_widths[col], std::string("BS").size() + 2);
        }
        col_widths[col++] = std::max(
                col_widths[col], std::to_string(config.m).size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], std::to_string(config.k).size() + 2);
        // N field (colon separated)
        std::string n_str;
        if (!config.n_values.empty()) {
            n_str += std::to_string(config.n_values[0]);
            for (size_t i = 1; i < config.n_values.size(); ++i) {
                n_str += ":" + std::to_string(config.n_values[i]);
            }
        }
        col_widths[col++] = std::max(col_widths[col], n_str.size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], std::to_string(config.iters).size() + 2);
        std::string dt_str = datatypeToStr(config.dt[0]) + ":"
                + datatypeToStr(config.dt[1]) + ":"
                + datatypeToStr(config.dt[2]);
        col_widths[col++] = std::max(col_widths[col], dt_str.size() + 2);
        col_widths[col++] = std::max(col_widths[col],
                std::to_string(config.isBiasEnabled).size() + 2);
        std::string bias_dt_str
                = config.isBiasEnabled ? datatypeToStr(config.bias_dt) : "";
        col_widths[col++] = std::max(col_widths[col], bias_dt_str.size() + 2);
        std::string postop_str;
        if (!config.post_ops.empty()) {
            postop_str += postOpsToStr(config.post_ops[0]);
            for (size_t j = 1; j < config.post_ops.size(); ++j) {
                postop_str += ":" + postOpsToStr(config.post_ops[j]);
            }
        }
        col_widths[col++] = std::max(col_widths[col], postop_str.size() + 2);
        col_widths[col++] = std::max(col_widths[col],
                config.binary_post_ops_pos.size() > 0
                        ? datatypeToStr(config.post_op_dt).size() + 2
                        : 0);
        col_widths[col++]
                = std::max(col_widths[col], config.kernel_name.size() + 2);
        col_widths[col++] = std::max(col_widths[col],
                std::to_string(config.is_weights_const).size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], std::to_string(config.isTransA).size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], std::to_string(config.isTransB).size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], std::to_string(config.alpha).size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], std::to_string(config.beta).size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], config.scale_granularity.size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], disp_wei_group_size(config).size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], disp_wei_scale_dt(config).size() + 2);
        col_widths[col++] = std::max(col_widths[col],
                std::to_string(config.warmup_iters).size() + 2);
        col_widths[col++] = std::max(col_widths[col],
                std::to_string(config.src_dynamic_quant).size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], disp_src_scale_granularity(config).size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], disp_src_group_size(config).size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], disp_src_scale_dt(config).size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], std::to_string((int)total_time).size() + 2);
        col_widths[col++] = std::max(col_widths[col],
                std::to_string((int)(total_time / config.iters)).size() + 2);
        col_widths[col++]
                = std::max(col_widths[col], std::string("GFLOPS").size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], std::string("%_of_Total").size() + 2);
#if MEASURE_INDIVIDUAL_TIMINGS
        col_widths[col++] = std::max(
                col_widths[col], std::string("Ctx_Creation(ms_%)").size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], std::string("Op_Creation(ms_%)").size() + 2);
        col_widths[col++] = std::max(
                col_widths[col], std::string("Op_Execution(ms_%)").size() + 2);
#endif
        // Per-layer rows
        for (auto i = 0; i < stat.size(); i++) {
            double percentage = (stat[i].total_time_ms / total_time) * 100;
            int st_index = 0;

            std::string layer_str = "Layer_" + std::to_string(i);
            col_widths[st_index]
                    = std::max(col_widths[st_index], layer_str.size() + 2);
            st_index++;
            if (inputMode == InputMode::MODEL) {
                col_widths[st_index] = std::max(col_widths[st_index],
                        std::string("Model Name").size() + 2);
                st_index++;
            }
            if (options.ndims > 2) {
                col_widths[st_index] = std::max(
                        col_widths[st_index], std::string("BS").size() + 2);
                st_index++;
            }
            cal_column_width(config, stat, col_widths, st_index, false, i,
                    percentage, true);
        }
    }

    // Helper lambda to print a row
    auto print_row = [&](const std::vector<std::string> &row) {
        for (size_t i = 0; i < row.size(); ++i) {
            outfile << std::setw(col_widths[i]) << row[i];
        }
        outfile << std::endl;
    };

    // Print table header
    outfile << std::fixed << std::setprecision(2);
    outfile << std::left;
    print_row(headers);

    // Print summary and per-layer rows for each configuration
    for (const auto &result : matmul_results) {
        const MatmulConfig &config = result.first;
        const std::vector<TimingStats> &stat = result.second;
        double total_time = 0.0;
        for (auto i = 0; i < config.n_values.size(); i++) {
            total_time += stat[i].total_time_ms;
        }
        // Summary row (aggregated for the pipeline)
        std::vector<std::string> summary_row;
        summary_row.push_back("Summary");
        if (inputMode == InputMode::MODEL) {
            summary_row.push_back(config.modelName);
        }
        if (options.ndims > 2) {
            summary_row.push_back(std::to_string(config.bs));
        }
        summary_row.push_back(std::to_string(config.m));
        summary_row.push_back(std::to_string(config.k));
        // N values as colon separated string
        std::string n_str;
        if (!config.n_values.empty()) {
            n_str += std::to_string(config.n_values[0]);
            for (size_t i = 1; i < config.n_values.size(); ++i) {
                n_str += ":" + std::to_string(config.n_values[i]);
            }
        }
        summary_row.push_back(n_str);
        summary_row.push_back(std::to_string(config.iters));
        summary_row.push_back(datatypeToStr(config.dt[0]) + ":"
                + datatypeToStr(config.dt[1]) + ":"
                + datatypeToStr(config.dt[2]));
        summary_row.push_back(std::to_string(config.isBiasEnabled));
        summary_row.push_back(
                config.isBiasEnabled ? datatypeToStr(config.bias_dt) : "");
        std::string postop_str;
        if (!config.post_ops.empty()) {
            postop_str += postOpsToStr(config.post_ops[0]);
            for (size_t j = 1; j < config.post_ops.size(); ++j) {
                postop_str += ":" + postOpsToStr(config.post_ops[j]);
            }
        }
        summary_row.push_back(postop_str);
        summary_row.push_back(config.binary_post_ops_pos.size() > 0
                        ? datatypeToStr(config.post_op_dt)
                        : "");
        summary_row.push_back(config.kernel_name);
        summary_row.push_back(std::to_string(config.is_weights_const));
        summary_row.push_back(std::to_string(config.isTransA));
        summary_row.push_back(std::to_string(config.isTransB));
        summary_row.push_back(std::to_string(config.alpha));
        summary_row.push_back(std::to_string(config.beta));
        summary_row.push_back(config.scale_granularity);
        summary_row.push_back(disp_wei_group_size(config));
        summary_row.push_back(disp_wei_scale_dt(config));
        summary_row.push_back(std::to_string(config.warmup_iters));
        summary_row.push_back(std::to_string(config.src_dynamic_quant));
        summary_row.push_back(disp_src_scale_granularity(config));
        summary_row.push_back(disp_src_group_size(config));
        summary_row.push_back(disp_src_scale_dt(config));
        std::ostringstream total_time_oss;
        total_time_oss << std::fixed << std::setprecision(2) << total_time;
        summary_row.push_back(total_time_oss.str());
        std::ostringstream avg_time_oss;
        avg_time_oss << std::fixed << std::setprecision(6)
                     << (total_time / config.iters);
        summary_row.push_back(avg_time_oss.str());
        summary_row.push_back("");
        summary_row.push_back("");
#if MEASURE_INDIVIDUAL_TIMINGS
        summary_row.push_back("");
        summary_row.push_back("");
        summary_row.push_back("");
#endif
        print_row(summary_row);

        // Per-layer rows (detailed timing for each layer in the pipeline)
        for (auto i = 0; i < stat.size(); i++) {
            double percentage = (stat[i].total_time_ms / total_time) * 100;
            std::vector<std::string> layer_row;
            layer_row.push_back("Layer_" + std::to_string(i));
            if (inputMode == InputMode::MODEL) {
                layer_row.push_back(config.modelName);
            }
            if (options.ndims > 2) {
                layer_row.push_back(std::to_string(config.bs));
            }
            fill_row(config, stat, layer_row, false, i, percentage, true);
            print_row(layer_row);
        }
    }
}

void log_results(std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>>
                         &matmul_results,
        std::ostream &outfile, const global_options &options,
        const bool isLOWOHA, const InputMode inputMode) {

    outfile << std::fixed << std::setprecision(2);
    if (inputMode == InputMode::MODEL) { outfile << "Model_Name, "; }
    if (options.ndims > 2) { outfile << "BS, "; }
    outfile << "M, K, N, Iterations, Data type, Bias Enabled, Bias Data type, "
            << "Post Operation, PostOp Data type, Kernel name, isWeightsConst, "
               "isTransA, isTransB, Alpha, Beta, "
            << "Weight Scale Granularity, Weight Group Size, Weight Scale Data "
               "type, Warmup iterations, "
            << "Src Dynamic Quant, Src Scale Granularity, Src Group Size, Src "
               "Scale Data type, "
            << "Total time (ms) (all iters),  Avg time (ms), GFLOPS";
#if MEASURE_INDIVIDUAL_TIMINGS
    if (!isLOWOHA) {
        outfile << ", Context Creation (ms & %), Operator Creation (ms & %), "
                   "Operator Execution (ms & %)";
    }
#endif
    outfile << std::endl;

    // Write results to CSV for each configuration
    for (const auto &result : matmul_results) {
        const MatmulConfig &config = result.first;
        const std::vector<TimingStats> &stat = result.second;
        if (inputMode == InputMode::MODEL) {
            outfile << config.modelName << ", ";
        }
        if (options.ndims > 2) { outfile << config.bs << ", "; }
        write_each_config_result(config, stat, outfile, isLOWOHA);
    }
}

void print_results(
        std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>>
                &matmul_results,
        std::ostream &outfile, const global_options &options,
        const bool isLOWOHA, const InputMode inputMode) {

    // Dynamic column widths calculation
    std::vector<std::string> headers;
    if (inputMode == InputMode::MODEL) {
        headers.insert(headers.begin(), "Model_Name");
    }
    if (options.ndims > 2) { headers.insert(headers.end(), "BS"); }
    headers.insert(headers.end(),
            {"M", "K", "N", "Iters", "Data_type", "Bias_Enabled", "Bias_dt",
                    "PostOp", "PostOp_dt", "Kernel_Name", "isWeightsConst",
                    "isTransA", "isTransB", "Alpha", "Beta",
                    "Weight_Scale_Granularity", "Weight_Group_Size",
                    "Weight_Scale_dt", "Warmup_iters", "Src_Dynamic_Quant",
                    "Src_Scale_Granularity", "Src_Group_Size", "Src_Scale_dt",
                    "Total_time(ms, all iters)", "Avg_time(ms)", "GFLOPS"});
#if MEASURE_INDIVIDUAL_TIMINGS
    if (!isLOWOHA) {
        headers.push_back("Ctx_Creation(ms_%)");
        headers.push_back("Op_Creation(ms_%)");
        headers.push_back("Op_Execution(ms_%)");
    }
#endif
    std::vector<size_t> col_widths(headers.size());
    // Initialize with header lengths
    for (size_t i = 0; i < headers.size(); ++i) {
        col_widths[i] = headers[i].size() + 2;
    }
    // Compute max width for each column based on all data rows
    for (const auto &result : matmul_results) {
        const MatmulConfig &config = result.first;
        const std::vector<TimingStats> &stat = result.second;
        // Column index offset for dynamic table formatting.
        int st_index = 0;
        if (inputMode == InputMode::MODEL) {
            col_widths[st_index] = std::max(
                    col_widths[st_index], config.modelName.size() + 2);
            st_index++;
        }
        // For BMM (ndims > 2), the first column is batch size (BS), so st_index is incremented.
        if (options.ndims > 2) {
            col_widths[st_index] = std::max(
                    col_widths[st_index], std::to_string(config.bs).size() + 2);
            st_index++;
        }
        cal_column_width(config, stat, col_widths, st_index, isLOWOHA);
    }

    // Helper lambda to print a row for the table
    auto print_row = [&](const std::vector<std::string> &row) {
        for (size_t i = 0; i < row.size(); ++i) {
            outfile << std::setw(col_widths[i]) << row[i];
        }
        outfile << std::endl;
    };

    // Print table header
    outfile << std::fixed << std::setprecision(2);
    outfile << std::left;
    print_row(headers);

    // Print each result row for every configuration
    for (const auto &result : matmul_results) {
        const MatmulConfig &config = result.first;
        const std::vector<TimingStats> &stat = result.second;
        std::vector<std::string> row;
        if (inputMode == InputMode::MODEL) { row.push_back(config.modelName); }
        if (options.ndims > 2) { row.push_back(std::to_string(config.bs)); }
        fill_row(config, stat, row, isLOWOHA);
        print_row(row);
    }
}

} // namespace matmul
} // namespace benchdnn
} // namespace zendnnl
