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

/**
 * @file ggml_weight_unpack.cpp
 * @brief GGML weight unpacking for ZenDNN kernels
 */

#include "ggml_weight_unpack.hpp"
#include "lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace {

struct block_q8_0  { uint16_t d; int8_t  qs[32];  };
struct block_q4_0  { uint16_t d; uint8_t qs[16];  };
struct block_q4_0x8 { uint16_t d[8]; uint8_t qs[128]; };

inline uint32_t fp32_to_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(f));
    return bits;
}

inline float fp32_from_bits(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(bits));
    return f;
}

// fp16 -> fp32, from llama.cpp
float fp16_to_fp32(uint16_t h) {
    const uint32_t w             = static_cast<uint32_t>(h) << 16;
    const uint32_t sign          = w & UINT32_C(0x80000000);
    const uint32_t two_w         = w + w;
    const uint32_t exp_offset    = UINT32_C(0xE0) << 23;
    const float    exp_scale     = fp32_from_bits(UINT32_C(0x7800000));
    const float    normalized    = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask    = UINT32_C(126) << 23;
    const float    denormalized  = fp32_from_bits((two_w >> 17) | magic_mask) - 0.5f;
    const uint32_t result = sign |
        (two_w < (UINT32_C(1) << 27) ? fp32_to_bits(denormalized)
                                     : fp32_to_bits(normalized));
    return fp32_from_bits(result);
}

// fp32 -> bf16, from llama.cpp
uint16_t fp32_to_bf16(float s) {
    union { float f; uint32_t i; } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000)
        return (u.i >> 16) | 64; // NaN: force quiet
    return (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
}

void write_scale(void *scale_buffer, bool use_bf16, int64_t idx, float val) {
    if (use_bf16)
        static_cast<uint16_t *>(scale_buffer)[idx] = fp32_to_bf16(val);
    else
        static_cast<float *>(scale_buffer)[idx] = val;
}

} // anonymous namespace

int64_t ggml_unpack_weight_buffer_size(int ggml_type, bool use_bf16_scales,
                                       int64_t M, int64_t K) {
    if (M <= 0 || K <= 0 || K % 32 != 0) return -1;
    int64_t ng          = K / 32;
    int64_t scale_bytes = M * ng * (use_bf16_scales ? 2 : 4);
    if (ggml_type == 8) return M * K       + scale_bytes;
    if (ggml_type == 2) return M * (K / 2) + scale_bytes;
    return -1;
}

int ggml_unpack_weight_buffer(const void *weight_data, int ggml_type,
                              bool is_superblock, bool use_bf16_scales,
                              bool use_unsigned_q4, int64_t M, int64_t K,
                              void *buf, int8_t **wei_ptr, void **scl_ptr) {
    if (!weight_data || !buf || !wei_ptr || !scl_ptr) return -1;
    if (M <= 0 || K <= 0 || K % 32 != 0) return -1;

    int64_t ng = K / 32;

    int64_t weight_bytes = (ggml_type == 8) ? M * K : M * (K / 2);
    *wei_ptr = static_cast<int8_t *>(buf);
    *scl_ptr = static_cast<void *>(static_cast<int8_t *>(buf) + weight_bytes);

    int8_t *weight_buffer = *wei_ptr;
    void   *scale_buffer  = *scl_ptr;

    // Q8_0: copy int8 weights directly, one scale per group.
    if (ggml_type == 8) {
        auto blocks = static_cast<const block_q8_0 *>(weight_data);
        for (int64_t row = 0; row < M; row++) {
            for (int64_t g = 0; g < ng; g++) {
                block_q8_0 local;
                std::memcpy(&local, &blocks[row * ng + g], sizeof(local));
                write_scale(scale_buffer, use_bf16_scales,
                            g * M + row, fp16_to_fp32(local.d));
                std::memcpy(&weight_buffer[row * K + g * 32], local.qs, 32);
            }
        }
        return 0;
    }

    // Q4_0 superblock: 8 rows interleaved, nibbles pre-signed via XOR 0x88.
    if (ggml_type == 2 && is_superblock) {
        if (M % 8 != 0) return -1;
        auto blocks = static_cast<const block_q4_0x8 *>(weight_data);
        for (int64_t r8 = 0; r8 < M / 8; r8++) {
            for (int64_t g = 0; g < ng; g++) {
                block_q4_0x8 local;
                std::memcpy(&local, &blocks[r8 * ng + g], sizeof(local));

                for (int ri = 0; ri < 8; ri++) {
                    int64_t row = r8 * 8 + ri;
                    write_scale(scale_buffer, use_bf16_scales,
                                g * M + row, fp16_to_fp32(local.d[ri]));

                    uint8_t src[16];
                    for (int i = 0; i < 8; i++) {
                        src[i]     = local.qs[ri * 8 + i];
                        src[i + 8] = local.qs[64 + ri * 8 + i];
                    }

                    int8_t *dst = &weight_buffer[(row * K + g * 32) / 2];
                    for (int i = 0; i < 8; i++) {
                        uint8_t lo0 = src[2*i]     & 0x0F;
                        uint8_t lo1 = src[2*i + 1] & 0x0F;
                        uint8_t hi0 = src[2*i]     >> 4;
                        uint8_t hi1 = src[2*i + 1] >> 4;

                        if (use_unsigned_q4) {
                            lo0 ^= 8; lo1 ^= 8; hi0 ^= 8; hi1 ^= 8;
                        }

                        dst[i]     = static_cast<int8_t>(lo0 | (lo1 << 4));
                        dst[8 + i] = static_cast<int8_t>(hi0 | (hi1 << 4));
                    }
                }
            }
        }
        return 0;
    }

    // Q4_0 regular: unsigned nibbles 0-15, subtract 8 to convert to signed S4.
    if (ggml_type == 2) {
        auto blocks = static_cast<const block_q4_0 *>(weight_data);
        for (int64_t row = 0; row < M; row++) {
            for (int64_t g = 0; g < ng; g++) {
                block_q4_0 local;
                std::memcpy(&local, &blocks[row * ng + g], sizeof(local));
                write_scale(scale_buffer, use_bf16_scales,
                            g * M + row, fp16_to_fp32(local.d));

                int8_t *dst = &weight_buffer[(row * K + g * 32) / 2];
                for (int i = 0; i < 8; i++) {
                    uint8_t lo0 = local.qs[2*i]     & 0x0F;
                    uint8_t lo1 = local.qs[2*i + 1] & 0x0F;
                    uint8_t hi0 = local.qs[2*i]     >> 4;
                    uint8_t hi1 = local.qs[2*i + 1] >> 4;

                    if (!use_unsigned_q4) {
                        lo0 = (lo0 - 8) & 0x0F;
                        lo1 = (lo1 - 8) & 0x0F;
                        hi0 = (hi0 - 8) & 0x0F;
                        hi1 = (hi1 - 8) & 0x0F;
                    }

                    dst[i]     = static_cast<int8_t>(lo0 | (lo1 << 4));
                    dst[8 + i] = static_cast<int8_t>(hi0 | (hi1 << 4));
                }
            }
        }
        return 0;
    }

    return -1; // unsupported type
}

status_t unpack_ggml_weights_and_cache(const void *&weight, int N, int K,
                                      matmul_params &params) {
  const int64_t unpack_M = static_cast<int64_t>(N);
  const int64_t unpack_K = static_cast<int64_t>(K);

  apilog_info("GGML Q8_0 unpack: N=", N, ", K=", K,
              ", weight_address=", static_cast<const void *>(weight));

  Key_unpack cache_key(weight,
                       static_cast<unsigned int>(N),
                       static_cast<unsigned int>(K));

  static lru_cache_t<Key_unpack, void *> unpack_cache;

  void *unpack_buf = nullptr;
  if (unpack_cache.find_key(cache_key)) {
    unpack_buf = unpack_cache.get(cache_key);
    apilog_info("GGML unpack cache hit: reading cached weights, N=", N,
                ", K=", K);
  } else {
    apilog_info("GGML unpack cache miss: unpacking and caching weights, N=", N,
                ", K=", K);

    int64_t buf_size = ggml_unpack_weight_buffer_size(
        8, true, unpack_M, unpack_K);
    if (buf_size <= 0) {
      log_error("Invalid parameters for GGML weight unpacking");
      return status_t::failure;
    }

    unpack_buf = std::malloc(static_cast<size_t>(buf_size));
    if (!unpack_buf) {
      log_error("Failed to allocate buffer for GGML weight unpacking");
      return status_t::failure;
    }

    int8_t *wei_tmp = nullptr;
    void *scl_tmp = nullptr;
    int ret = ggml_unpack_weight_buffer(
        weight, 8, false, true, false,
        unpack_M, unpack_K, unpack_buf, &wei_tmp, &scl_tmp);
    if (ret != 0) {
      std::free(unpack_buf);
      log_error("GGML weight unpacking failed");
      return status_t::failure;
    }

    unpack_cache.add(cache_key, unpack_buf);
    apilog_info("GGML unpack complete: wrote ", buf_size,
                " bytes (weights + bf16 scales)");
  }

  int64_t weight_bytes = unpack_M * unpack_K;
  weight = static_cast<const void *>(unpack_buf);

  params.quant_params.wei_scale.buff = static_cast<const void *>(
      static_cast<int8_t *>(unpack_buf) + weight_bytes);
  params.quant_params.wei_scale.dt = data_type_t::bf16;
  int64_t ng = unpack_K / 32;
  params.quant_params.wei_scale.dims = {ng, unpack_M};

  apilog_info("GGML unpack output: weight_bytes=", weight_bytes,
              ", scale_groups=", ng, ", scale_dt=bf16");

  return status_t::success;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
