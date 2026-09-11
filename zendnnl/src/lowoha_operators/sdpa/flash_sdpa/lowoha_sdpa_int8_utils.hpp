/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# *******************************************************************************/

#ifndef LOWOHA_SDPA_INT8_UTILS_HPP
#define LOWOHA_SDPA_INT8_UTILS_HPP

#include <cstdint>
#include <vector>

#include "lowoha_operators/sdpa/lowoha_sdpa_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

// These helpers adapt SDPA tile/head metadata to LOWOHA quantization and
// matmul operations.

// Q/K quantization: one scale per row, so `scales[i]` holds `rows[i]` floats.
status_t sdpa_group_quantize_bf16_s8_per_token(
        const std::vector<const void *> &src, const std::vector<int> &rows,
        const std::vector<int> &cols,
        const std::vector<std::vector<int64_t>> &src_strides,
        const std::vector<void *> &dst, const std::vector<void *> &scales,
        int num_threads);

// V quantization: one scale per column, so `scales[i]` holds `cols[i]` floats.
status_t sdpa_group_quantize_bf16_s8_per_channel(
        const std::vector<const void *> &src, const std::vector<int> &rows,
        const std::vector<int> &cols,
        const std::vector<std::vector<int64_t>> &src_strides,
        const std::vector<void *> &dst, const std::vector<void *> &scales,
        int num_threads);

status_t sdpa_qk_int8_matmul(const int8_t *query, const int8_t *key,
        float *scores, int M, int N, int K, const float *query_scales,
        const float *key_scales);

status_t sdpa_pv_int8_matmul(const uint8_t *probability, const int8_t *value,
        float *output, int M, int N, int K, const float *probability_scale,
        const float *value_scales);

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_SDPA_INT8_UTILS_HPP
