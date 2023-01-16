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

#ifndef ZENDNN_HW_OS_KERNEL_BIOS_INFO_HPP
#define ZENDNN_HW_OS_KERNEL_BIOS_INFO_HPP

#include <stdio.h>
#include <string>

namespace zendnn {

class zendnnHwOsKernelBiosEnv {
    public:
        static zendnnHwOsKernelBiosEnv* Instance();
        void readHwEnv();
        void readOsEnv();
        void readKernelEnv();
        void readBiosEnv();

        //HW environment
        std::string hw_architecture;
        std::string hw_cpu_op_mode;
        std::string hw_byte_order;
        std::string hw_addrs_size;
        unsigned int hw_num_threads;
        std::string hw_cpu_list;
        unsigned int hw_thread_core;
        unsigned int hw_cores_socket;
        unsigned int hw_num_sockets;
        unsigned int hw_numa_node_socket;
        unsigned int hw_vendor_id;
        unsigned int hw_cpu_family;
        unsigned int hw_model;
        std::string hw_model_name;
        unsigned int hw_stepping;
        std::string hw_freq_boost;
        float hw_cpu_mhz;
        float hw_cpu_max_mhz;
        float hw_cpu_min_mhz;
        float hw_bogomips;
        std::string hw_virtualization;
        std::string hw_l1d_cache_core;
        std::string hw_l1i_cache_core;
        std::string hw_l2_cache_core;
        std::string hw_l3_cache_ccx_ccd;
        unsigned int hw_cores_ccx;
        std::string hw_equi_l3_cache_core;

        //OS environment details
        std::string os_name;
        std::string os_id_like;
        std::string os_version;
        std::string os_version_id;

        //Kernel environment details
        std::string ker_kernel_name;
        std::string ker_node_hostname;
        std::string ker_kernel_rel;
        std::string ker_kernel_ver;
        std::string ker_machine_hw_name;
        std::string ker_os_name;

        //BIOS environment details
        std::string bios_version;
        std::string bios_vendor;
        std::string bios_release_date;
        std::string bios_proc_family;

        //FIXME:
        unsigned int omp_num_threads;

    private:
        zendnnHwOsKernelBiosEnv() {};    //Private so that it cannot be called
        zendnnHwOsKernelBiosEnv(zendnnHwOsKernelBiosEnv const&); //copy constructor is private, dont implement
        void operator=(zendnnHwOsKernelBiosEnv const&); //assignmnet operator is private; dont implement
        static zendnnHwOsKernelBiosEnv* m_pInstance;

};

} //namespace zendnn
#endif //ZENDNN_HW_OS_KERNEL_BIOS_INFO_HPP
