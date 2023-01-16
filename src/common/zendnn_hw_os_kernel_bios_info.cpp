/*******************************************************************************
* Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
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
*
*******************************************************************************/

#include <stddef.h>
#include <iostream>
#include <string>
#include "zendnn_hw_os_kernel_bios_info.hpp"
#include "zendnn_helper.hpp"
#include "zendnn_logging.hpp"

using namespace zendnn;

//Global static pointer to ensure a single instance of the class
zendnnHwOsKernelBiosEnv* zendnnHwOsKernelBiosEnv::m_pInstance = NULL;

/* This function is called to create an instance of the class
 * Calling the constructor publicaly is not allowed.
 * The constructor is private and is only called by this Instance function
 */

zendnnHwOsKernelBiosEnv* zendnnHwOsKernelBiosEnv::Instance() {
    if(!m_pInstance) //only allow one instance of the class to be generated
        m_pInstance = new zendnnHwOsKernelBiosEnv;

    return m_pInstance;
}

void zendnnHwOsKernelBiosEnv::readHwEnv() {
    hw_architecture = zendnn_getenv_string("_SYSTEM_HW_ARCHITECTURE");
    hw_cpu_op_mode = zendnn_getenv_string("_SYSTEM_HW_CPU_OP_MODE");
    hw_byte_order = zendnn_getenv_string("_SYSTEM_HW_BYTE_ORDER");
    hw_addrs_size = zendnn_getenv_string("_SYSTEM_HW_ADDRS_SIZE");
    hw_num_threads = zendnn_getenv_int("_SYSTEM_HW_NUM_THREADS");
    hw_cpu_list = zendnn_getenv_string("_SYSTEM_HW_CPU_LIST");
    hw_thread_core = zendnn_getenv_int("_SYSTEM_HW_THEARD_CORE");
    hw_cores_socket = zendnn_getenv_int("_SYSTEM_HW_CORES_SOCKET");
    hw_num_sockets = zendnn_getenv_int("_SYSTEM_HW_NUM_SOCKETS");
    hw_numa_node_socket = zendnn_getenv_int("_SYSTEM_HW_NUMA_NODE_SOCKET");
    hw_vendor_id = zendnn_getenv_int("_SYSTEM_HW_VENDOR_ID");
    hw_cpu_family = zendnn_getenv_int("_SYSTEM_HW_CPU_FAMILY");
    hw_model = zendnn_getenv_int("_SYSTEM_HW_MODEL");
    hw_model_name = zendnn_getenv_string("_SYSTEM_HW_MODEL_NAME");
    hw_stepping = zendnn_getenv_int("_SYSTEM_HW_STEPPING");
    hw_freq_boost = zendnn_getenv_string("_SYSTEM_HW_FREQ_BOOST");
    hw_cpu_mhz = zendnn_getenv_float("_SYSTEM_HW_CPU_MHZ");
    hw_cpu_max_mhz = zendnn_getenv_float("_SYSTEM_HW_CPU_MAX_MHZ");
    hw_cpu_min_mhz = zendnn_getenv_float("_SYSTEM_HW_CPU_MIN_MHZ");
    hw_bogomips = zendnn_getenv_float("_SYSTEM_HW_BOGOMIPS");
    hw_virtualization = zendnn_getenv_string("_SYSTEM_HW_VIRTUALIZATION");
    hw_l1d_cache_core = zendnn_getenv_string("_SYSTEM_HW_L1D_CACHE_CORE");
    hw_l1i_cache_core = zendnn_getenv_string("_SYSTEM_HW_L1I_CACHE_CORE");
    hw_l2_cache_core = zendnn_getenv_string("_SYSTEM_HW_L2_CACHE_CORE");
    hw_l3_cache_ccx_ccd = zendnn_getenv_string("_SYSTEM_HW_L3_CACHE_CCX_CCD");
    hw_cores_ccx = zendnn_getenv_int("_SYSTEM_HW_CORES_CCX");
    hw_equi_l3_cache_core = zendnn_getenv_string("_SYSTEM_HW_EQUI_L3_CACHE_CORE");
}

void zendnnHwOsKernelBiosEnv::readOsEnv() {
    os_name = zendnn_getenv_string("_SYSTEM_OS_ID");
    os_id_like = zendnn_getenv_string("_SYSTEM_OS_ID_LIKE");
    os_version = zendnn_getenv_string("_SYSTEM_OS_VERSION");
    os_version_id = zendnn_getenv_string("_SYSTEM_OS_VERSION_ID");
}

void zendnnHwOsKernelBiosEnv::readKernelEnv() {
    ker_kernel_name = zendnn_getenv_string("_SYSTEM_KER_KERNEL_NAME");
    ker_node_hostname = zendnn_getenv_string("_SYSTEM_KER_NODE_HOSTNAME");
    ker_kernel_rel = zendnn_getenv_string("_SYSTEM_KER_KERNEL_REL");
    ker_kernel_ver = zendnn_getenv_string("_SYSTEM_KER_KERNEL_VER");
    ker_machine_hw_name = zendnn_getenv_string("_SYSTEM_KER_MACHINE_HW_NAME");
    ker_os_name = zendnn_getenv_string("_SYSTEM_KER_OS_NAME");
}

void zendnnHwOsKernelBiosEnv::readBiosEnv() {
    bios_version = zendnn_getenv_string("_SYSTEM_BIOS_VERSION");
    bios_vendor = zendnn_getenv_string("_SYSTEM_BIOS_VENDOR");
    bios_release_date = zendnn_getenv_string("_SYSTEM_BIOS_RELEASE_DATE");
    bios_proc_family = zendnn_getenv_string("_SYSTEM_BIOS_PROC_FAMILY");
}

