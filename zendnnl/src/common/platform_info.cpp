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

#include "platform_info.hpp"

#include <cstdint>
#ifdef __x86_64__
  #include <cpuid.h>
#endif

namespace zendnnl {
namespace common {

using namespace Au;

platform_info_t::platform_info_t()
  : is_avx2{false}, is_avx512f{false}, is_avx512f16{false}, isa_version{0},
    cpu_family{0}, cpu_model{0}, cpu_vendor{}, cpu_uarch{0} {
}

status_t platform_info_t::populate() {
  X86Cpu cpu{0};

  auto v_info     = cpu.getVendorInfo();
  cpu_vendor      = static_cast<uint32_t>(v_info.m_mfg);
  cpu_family      = static_cast<uint32_t>(v_info.m_family);
  cpu_model       = v_info.m_model;
  cpu_uarch       = static_cast<uint32_t>(v_info.m_uarch);
  is_avx2         = cpu.hasFlag(ECpuidFlag::avx2);
  is_avx512f      = cpu.hasFlag(ECpuidFlag::avx512f);
  is_avx512f16    = detect_f16_isa();

  return status_t::success;
}

bool platform_info_t::detect_f16_isa() {
#ifdef __x86_64__
  uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;

  // Check 1: AVX512-FP16 (CPUID leaf 7, subleaf 0, EDX bit 23)
  // Covers: avx512_core_fp16
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  if ((edx >> 23) & 1) {
    return true;
  }

  // Check 2: AVX-NE-CONVERT (CPUID leaf 7, subleaf 1, EDX bit 5)
  // Covers: avx2_vnni_2
  eax = ebx = ecx = edx = 0;
  __cpuid_count(7, 1, eax, ebx, ecx, edx);
  if ((edx >> 5) & 1) {
    return true;
  }
#endif
  return false;
}

bool platform_info_t::get_avx2_status() const {
  return is_avx2;
}

bool platform_info_t::get_avx512f_status() const {
  return is_avx512f;
}

bool platform_info_t::get_f16_status() const {
  return is_avx512f16;
}

uint32_t platform_info_t::get_isa_version() const {
  return isa_version;
}

uint32_t platform_info_t::get_cpu_family() const {
  return cpu_family;
}

uint32_t platform_info_t::get_cpu_model() const {
  return cpu_model;
}

uint32_t platform_info_t::get_cpu_vendor() const {
  return cpu_vendor;
}

uint32_t platform_info_t::get_cpu_uarch() const {
  return cpu_uarch;
}

}//common
}//zendnnl
