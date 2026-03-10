/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "common/platform_info.hpp"
#include "common/zendnnl_global.hpp"

#include <cpuid.h>
#include <mutex>
#include <omp.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

static int read_amd_cache_size(int level) {
  for (unsigned sub = 0; sub < 16; ++sub) {
    unsigned eax = 0, ebx = 0, ecx_out = 0, edx = 0;
    __cpuid_count(0x8000001Du, sub, eax, ebx, ecx_out, edx);
    int cache_type = eax & 0x1F;
    if (cache_type == 0) break;
    int cache_level = (eax >> 5) & 0x7;
    if (cache_level == level && (cache_type == 1 || cache_type == 3)) {
      int line_size  = (ebx & 0xFFF) + 1;
      int partitions = ((ebx >> 12) & 0x3FF) + 1;
      int ways     = ((ebx >> 22) & 0x3FF) + 1;
      int sets     = ecx_out + 1;
      return static_cast<int>(static_cast<int64_t>(line_size) * partitions * ways * sets);
    }
  }
  return 0;
}

static int read_intel_cache_size(int level) {
  for (unsigned sub = 0; sub < 16; ++sub) {
    unsigned eax = 0, ebx = 0, ecx_out = 0, edx = 0;
    __cpuid_count(0x04u, sub, eax, ebx, ecx_out, edx);
    int cache_type = eax & 0x1F;
    if (cache_type == 0) break;
    int cache_level = (eax >> 5) & 0x7;
    if (cache_level == level && (cache_type == 1 || cache_type == 3)) {
      int line_size  = (ebx & 0xFFF) + 1;
      int partitions = ((ebx >> 12) & 0x3FF) + 1;
      int ways     = ((ebx >> 22) & 0x3FF) + 1;
      int sets     = ecx_out + 1;
      return static_cast<int>(static_cast<int64_t>(line_size) * partitions * ways * sets);
    }
  }
  return 0;
}

static UarchParams do_detect() {
  UarchParams p;
  auto &pinfo = zendnnl::common::zendnnl_platform_info();
  p.avx512f   = pinfo.get_avx512f_status();
  // Note: num_cores is the OMP thread count, not physical core count.
  p.num_cores = omp_get_max_threads();

  uint32_t cpu_family = pinfo.get_cpu_family();
  [[maybe_unused]] uint32_t cpu_model = pinfo.get_cpu_model();

  unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
  __cpuid(0, eax, ebx, ecx, edx);
  bool is_amd = (ebx == 0x68747541);

  if (is_amd) {
    int l1 = read_amd_cache_size(1);
    int l2 = read_amd_cache_size(2);
    int l3 = read_amd_cache_size(3);
    p.l1d_bytes = l1 > 0 ? l1 : 32768;
    p.l2_bytes  = l2 > 0 ? l2 : 1048576;
    p.l3_bytes_per_ccd = l3 > 0 ? l3 : 33554432;
  } else {
    int l1 = read_intel_cache_size(1);
    int l2 = read_intel_cache_size(2);
    int l3 = read_intel_cache_size(3);
    p.l1d_bytes = l1 > 0 ? l1 : 32768;
    p.l2_bytes  = l2 > 0 ? l2 : 1048576;
    p.l3_bytes_per_ccd = l3 > 0 ? l3 : 33554432;
  }

  p.fma_ports = 2;
  if (is_amd && cpu_family == 0x1A) {
    p.vec_width_bits = 512;
    if (p.l1d_bytes < 49152) p.l1d_bytes = 49152;
  } else if (is_amd && cpu_family == 0x19) {
    p.vec_width_bits = 256;
  } else {
    p.vec_width_bits = p.avx512f ? 512 : 256;
  }

  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  p.avx512vnni = p.avx512f && (ecx & (1u << 11));
  __cpuid_count(7, 1, eax, ebx, ecx, edx);
  p.avx512bf16 = p.avx512f && (eax & (1u << 5));

  return p;
}

const UarchParams &detect_uarch() {
  static UarchParams cached = do_detect();
  return cached;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
