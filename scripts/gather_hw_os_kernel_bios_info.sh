#*******************************************************************************
# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#*******************************************************************************

#!/bin/bash

#-----------------------------------------------------------------------------
#   gather_hw_os_kernel_bios_info.sh
#   Prerequisite: This script requires that lscpu, lstopo, uname, dmidecode
#                 is installed in the system.
#
#   This script does following steps:
#   -Gather information on
#       -H/W
#       -OS
#       -Kernel
#       -BIOS
#   - Prints the information if argument is true
#
#   Usage:
#   source scripts/gather_hw_os_kernel_bios_info.sh true/false
#
#----------------------------------------------------------------------------


# Enable/Disable HW_OS_KERNEL_BIOS environment export prints
_HW_OS_KERNEL_BIOS_EXPORT_PRINT=${1:-false}

#-------------------------------------------------------------------------------
# HW, HW architecture, Cache, OS, Kernel details
#-----------------------------------------------------------------------------

#Function to check optional prerequisites
function check_optional_prereqs() {
    if type -P lscpu >/dev/null 2>&1;
    then
        echo "lscpu is installed"
        echo `lscpu --version`
    else
        echo "lscpu is not installed, install lscpu"
    fi

    # Check if hwloc/lstopo-no-graphics is installed
    if type -P lstopo-no-graphics >/dev/null 2>&1;
    then
        echo "lstopo-no-graphics is installed"
        echo `lstopo-no-graphics --version`
    else
        echo "lstopo-no-graphics is not installed, install hwloc/lstopo-no-graphics"
    fi

    # Check if uname is installed
    if type -P uname >/dev/null 2>&1;
    then
        echo "uname is installed"
        echo `uname --version`
    else
        echo "uname is not installed, install uname"
    fi

    # Check if dmidecode is installed
    if type -P dmidecode >/dev/null 2>&1;
    then
        echo "dmidecode is installed"
        echo `dmidecode --version`
    else
        echo "dmidecode is not installed, install dmidecode"
    fi
}

#------------------------------------------------------------------------------
echo "Checking optional prerequisites"
# Check if optional prerequisites are installed
check_optional_prereqs
echo""

#------------------------------------------------------------------------------
# Print HW details
export _SYSTEM_HW_ARCHITECTURE=$(lscpu | grep 'Architecture' | awk '{print $2}')
export _SYSTEM_HW_CPU_OP_MODE=$(lscpu | grep 'CPU op-mode(s)' | awk '{print $3,$4,$5}')
export _SYSTEM_HW_BYTE_ORDER=$(lscpu | grep 'Byte Order:' | awk '{print $3,$4}')
export _SYSTEM_HW_ADDRS_SIZE=$(lscpu | grep 'Address sizes' -m1 | awk '{print $2,$3,$4,$5,$6,$7,$8}')
export _SYSTEM_HW_NUM_THREADS=$(lscpu | grep 'CPU(s)' -m1 | awk '{print $2}')
export _SYSTEM_HW_CPU_LIST=$(lscpu | grep 'On-line CPU(s) list' | awk '{print $4}')
export _SYSTEM_HW_THEARD_CORE=$(lscpu | grep 'Thread(s) per core' | awk '{print $4}')
export _SYSTEM_HW_CORES_SOCKET=$(lscpu | grep 'Core(s) per socket' | awk '{print $4}')
export _SYSTEM_HW_NUM_SOCKETS=$(lscpu | grep 'Socket(s)' | awk '{print $2}')
export _SYSTEM_HW_NUMA_NODES=$(lscpu | grep 'NUMA node(s)' | awk '{print $3}')
export _SYSTEM_HW_NUMA_NODE_SOCKET=$((_SYSTEM_HW_NUMA_NODES/_SYSTEM_HW_NUM_SOCKETS))
export _SYSTEM_HW_VENDOR_ID=$(lscpu | grep 'NUMA node(s)' | awk '{print $3}')
export _SYSTEM_HW_CPU_FAMILY=$(lscpu | grep 'CPU family' | awk '{print $3}')
export _SYSTEM_HW_MODEL=$(lscpu | grep 'Model' -m1 | awk '{print $2}')
export _SYSTEM_HW_MODEL_NAME=$(lscpu | grep 'Model name:' | awk '{print $3,$4,$5,$6,$7,$8}')
export _SYSTEM_HW_STEPPING=$(lscpu | grep 'Stepping' | awk '{print $2}')
export _SYSTEM_HW_FREQ_BOOST=$(lscpu | grep 'Frequency boost:' | awk '{print $3}')
export _SYSTEM_HW_CPU_MHZ=$(lscpu | grep 'CPU MHz' | awk '{print $3}')
export _SYSTEM_HW_CPU_MAX_MHZ=$(lscpu | grep 'CPU max MHz' | awk '{print $4}')
export _SYSTEM_HW_CPU_MIN_MHZ=$(lscpu | grep 'CPU min MHz' | awk '{print $4}')
export _SYSTEM_HW_BOGOMIPS=$(lscpu | grep 'BogoMIPS' | awk '{print $2}')
export _SYSTEM_HW_VIRTUALIZATION=$(lscpu | grep 'Virtualization' | awk '{print $2}')
export _SYSTEM_HW_L1D_CACHE_CORE=$(lstopo-no-graphics | grep ' L2 L#0' |awk '{print $7}' | sed 's/[()]//g')
export _SYSTEM_HW_L1I_CACHE_CORE=$(lstopo-no-graphics | grep ' L2 L#0' |awk '{print $11}'| sed 's/[()]//g')
export _SYSTEM_HW_L2_CACHE_CORE=$(lstopo-no-graphics | grep ' L2 L#0' |awk '{print $3}' | sed 's/[()]//g')
export _SYSTEM_HW_L3_CACHE_CCX_CCD=$(lstopo-no-graphics | grep ' L3 L#0' |awk '{print $3}' | sed 's/[()]//g')
#FIXME: Flags => sse sse2 ssse3 sse4_1 sse4_2 avx sse4a avx2
export _SYSTEM_HW_FLAGS=$(lscpu | grep 'Flags' | grep -o 'sse\|sse2\|ssse3\|sse4_1\|sse4_2\|avx\|sse4a\|avx2\|avx512')

if [[ "$_HW_OS_KERNEL_BIOS_EXPORT_PRINT" = true ]]; then
    echo ""
    echo "HW Details:"
    echo "_SYSTEM_HW_ARCHITECTURE: "$_SYSTEM_HW_ARCHITECTURE
    echo "_SYSTEM_HW_CPU_OP_MODE: "$_SYSTEM_HW_CPU_OP_MODE
    echo "_SYSTEM_HW_BYTE_ORDER: "$_SYSTEM_HW_BYTE_ORDER
    echo "_SYSTEM_HW_ADDRS_SIZE: "$_SYSTEM_HW_ADDRS_SIZE
    echo "_SYSTEM_HW_NUM_THREADS: "$_SYSTEM_HW_NUM_THREADS
    echo "_SYSTEM_HW_CPU_LIST: "$_SYSTEM_HW_CPU_LIST
    echo "_SYSTEM_HW_THEARD_CORE (SMT): "$_SYSTEM_HW_THEARD_CORE
    echo "_SYSTEM_HW_CORES_SOCKET: "$_SYSTEM_HW_CORES_SOCKET
    echo "_SYSTEM_HW_NUM_SOCKETS (1P/2P): "$_SYSTEM_HW_NUM_SOCKETS
    #echo "_SYSTEM_HW_NUMA_NODES: "$_SYSTEM_HW_NUMA_NODES
    echo "_SYSTEM_HW_NUMA_NODE_SOCKET: "$_SYSTEM_HW_NUMA_NODE_SOCKET
    echo "_SYSTEM_HW_VENDOR_ID: "$_SYSTEM_HW_VENDOR_ID
    echo "_SYSTEM_HW_CPU_FAMILY: "$_SYSTEM_HW_CPU_FAMILY
    echo "_SYSTEM_HW_MODEL: "$_SYSTEM_HW_MODEL
    echo "_SYSTEM_HW_MODEL_NAME: "$_SYSTEM_HW_MODEL_NAME
    echo "_SYSTEM_HW_STEPPING: "$_SYSTEM_HW_STEPPING
    echo "_SYSTEM_HW_FREQ_BOOST: "$_SYSTEM_HW_FREQ_BOOST
    echo "_SYSTEM_HW_CPU_MHZ: "$_SYSTEM_HW_CPU_MHZ
    echo "_SYSTEM_HW_CPU_MAX_MHZ: "$_SYSTEM_HW_CPU_MAX_MHZ
    echo "_SYSTEM_HW_CPU_MIN_MHZ: "$_SYSTEM_HW_CPU_MIN_MHZ
    echo "_SYSTEM_HW_BOGOMIPS: "$_SYSTEM_HW_BOGOMIPS
    echo "_SYSTEM_HW_VIRTUALIZATION: "$_SYSTEM_HW_VIRTUALIZATION
    echo "_SYSTEM_HW_L1D_CACHE_CORE: "$_SYSTEM_HW_L1D_CACHE_CORE
    echo "_SYSTEM_HW_L1I_CACHE_CORE: "$_SYSTEM_HW_L1I_CACHE_CORE
    echo "_SYSTEM_HW_L2_CACHE_CORE: "$_SYSTEM_HW_L2_CACHE_CORE
    echo "_SYSTEM_HW_L3_CACHE_CCX_CCD: "$_SYSTEM_HW_L3_CACHE_CCX_CCD
    echo "_SYSTEM_HW_FLAGS: "$_SYSTEM_HW_FLAGS
fi

_HW_LSTOPO_PACKAGES=$(lstopo-no-graphics -s | grep 'Package' | awk '{print $3}')
_HW_LSTOPO_NUM_L3CACHE=$(lstopo-no-graphics -s | grep 'L3Cache' | awk '{print $3}')
_HW_LSTOPO_NUM_L2CACHE=$(lstopo-no-graphics -s | grep 'L2Cache' | awk '{print $3}')
_HW_NUM_CORES_SOCKET=$((_HW_LSTOPO_NUM_L2CACHE/_HW_LSTOPO_PACKAGES))
_HW_LSTOPO_NUM_L3_CACHE_SOCKET=$((_HW_LSTOPO_NUM_L3CACHE/_HW_LSTOPO_PACKAGES))
_HW_PRCOCESSOR_NUM=$(lscpu | grep 'Model name:' | awk '{print $5}')

#if [[ "$_HW_OS_KERNEL_BIOS_EXPORT_PRINT" = true ]]; then
    #echo "_HW_LSTOPO_PACKAGES: "$_HW_LSTOPO_PACKAGES
    #echo "_HW_LSTOPO_NUM_L3CACHE: "$_HW_LSTOPO_NUM_L3CACHE
    #echo "_HW_LSTOPO_NUM_L2CACHE: "$_HW_LSTOPO_NUM_L2CACHE
    #echo "_HW_NUM_CORES_SOCKET: "$_HW_NUM_CORES_SOCKET
    #echo "_HW_LSTOPO_NUM_L3_CACHE_SOCKET: "$_HW_LSTOPO_NUM_L3_CACHE_SOCKET
    #echo "_HW_PRCOCESSOR_NUM: "$_HW_PRCOCESSOR_NUM
#fi

#Find the last digit of EPYC model number
EPYC_FAMILY_LAST_DIGIT=$(cat /proc/cpuinfo | grep 'model name' -m1 | awk '{print substr($6, 4);}')
#echo $EPYC_FAMILY_LAST_DIGIT

if [[ " $EPYC_FAMILY_LAST_DIGIT " == " 2 " ]]; then
    export _SYSTEM_HW_CORES_CCX=$((_HW_NUM_CORES_SOCKET/_HW_LSTOPO_NUM_L3_CACHE_SOCKET))
    _L3CACHE_SIZE_CCX_NUM=$(echo $_SYSTEM_HW_L3_CACHE_CCX_CCD | tr -dc '0-9')
    _L3CACHE_SIZE_CCX_UNIT=${_SYSTEM_HW_L3_CACHE_CCX_CCD//[0-9]/}
    _SYSTEM_HW_EQUI_L3_CACHE_VAL_CORE=$(echo $(( 100 * $_L3CACHE_SIZE_CCX_NUM / $_SYSTEM_HW_CORES_CCX )) | sed 's/..$/.&/')
    #echo $_SYSTEM_HW_EQUI_L3_CACHE_VAL_CORE
    export _SYSTEM_HW_EQUI_L3_CACHE_CORE="${_SYSTEM_HW_EQUI_L3_CACHE_VAL_CORE}${_L3CACHE_SIZE_CCX_UNIT}"
    if [[ "$_HW_OS_KERNEL_BIOS_EXPORT_PRINT" = true ]]; then
        echo "This is Rome "$_HW_PRCOCESSOR_NUM
        echo "_SYSTEM_HW_CORES_CCX: "$_SYSTEM_HW_CORES_CCX
        echo "_SYSTEM_HW_EQUI_L3_CACHE_CORE: "$_SYSTEM_HW_EQUI_L3_CACHE_CORE
    fi
elif [[ " $EPYC_FAMILY_LAST_DIGIT " == " 3 " ]]; then
    export _SYSTEM_HW_CORES_CCD=$((_HW_NUM_CORES_SOCKET/_HW_LSTOPO_NUM_L3_CACHE_SOCKET))
    _L3CACHE_SIZE_CCD_NUM=$(echo $_SYSTEM_HW_L3_CACHE_CCX_CCD | tr -dc '0-9')
    _L3CACHE_SIZE_CCD_UNIT=${_SYSTEM_HW_L3_CACHE_CCX_CCD//[0-9]/}
    _SYSTEM_HW_EQUI_L3_CACHE_VAL_CORE=$(echo $(( 100 * $_L3CACHE_SIZE_CCD_NUM / $_SYSTEM_HW_CORES_CCD )) | sed 's/..$/.&/')
    #echo $_SYSTEM_HW_EQUI_L3_CACHE_VAL_CORE
    export _SYSTEM_HW_EQUI_L3_CACHE_CORE="${_SYSTEM_HW_EQUI_L3_CACHE_VAL_CORE}${_L3CACHE_SIZE_CCD_UNIT}"
    if [[ "$_HW_OS_KERNEL_BIOS_EXPORT_PRINT" = true ]]; then
        echo "This is Milan "$_HW_PRCOCESSOR_NUM
        echo "_SYSTEM_HW_CORES_CCD: "$_SYSTEM_HW_CORES_CCD
        echo "_SYSTEM_HW_EQUI_L3_CACHE_CORE: "$_SYSTEM_HW_EQUI_L3_CACHE_CORE
    fi
else
    if [[ "$_HW_OS_KERNEL_BIOS_EXPORT_PRINT" = true ]]; then
        echo "Unable to determine CPU Architecture"
        echo "Env variable _SYSTEM_HW_CORES_CCD is not defined"
        echo "Env variable _SYSTEM_HW_EQUI_L3_CACHE_CORE is not defined"
    fi
fi
#FIXME: Information about number of memory channel

# Print OS details
export _SYSTEM_OS_ID=$(cat /etc/os-release | grep 'ID' -m1 | awk -F= '{print $2}')
export _SYSTEM_OS_ID_LIKE=$(cat /etc/os-release | grep 'ID_LIKE' -m1 | awk -F= '{print $2}')
export _SYSTEM_OS_VERSION=$(cat /etc/os-release | grep 'VERSION' -m1 | awk -F= '{print $2}')
export _SYSTEM_OS_VERSION_ID=$(cat /etc/os-release | grep 'VERSION_ID' -m1 | awk -F= '{print $2}')

if [[ "$_HW_OS_KERNEL_BIOS_EXPORT_PRINT" = true ]]; then
    echo ""
    echo "OS Details:"
    echo "_SYSTEM_OS_ID: "$_SYSTEM_OS_ID
    echo "_SYSTEM_OS_ID_LIKE: "$_SYSTEM_OS_ID_LIKE
    echo "_SYSTEM_OS_VERSION: "$_SYSTEM_OS_VERSION
    echo "_SYSTEM_OS_VERSION_ID: "$_SYSTEM_OS_VERSION_ID
fi

# Print Kernel details
export _SYSTEM_KER_KERNEL_NAME=$(uname -s)
export _SYSTEM_KER_NODE_HOSTNAME=$(uname -n)
export _SYSTEM_KER_KERNEL_REL=$(uname -r)
export _SYSTEM_KER_KERNEL_VER=$(uname -v)
export _SYSTEM_KER_MACHINE_HW_NAME=$(uname -m)
export _SYSTEM_KER_OS_NAME=$(uname -o)

if [[ "$_HW_OS_KERNEL_BIOS_EXPORT_PRINT" = true ]]; then
    echo ""
    echo "Kernel Details:"
    echo "_SYSTEM_KER_KERNEL_NAME: "$_SYSTEM_KER_KERNEL_NAME
    echo "_SYSTEM_KER_NODE_HOSTNAME: "$_SYSTEM_KER_NODE_HOSTNAME
    echo "_SYSTEM_KER_KERNEL_REL: "$_SYSTEM_KER_KERNEL_REL
    echo "_SYSTEM_KER_KERNEL_VER: "$_SYSTEM_KER_KERNEL_VER
    echo "_SYSTEM_KER_MACHINE_HW_NAME: "$_SYSTEM_KER_MACHINE_HW_NAME
    echo "_SYSTEM_KER_OS_NAME: "$_SYSTEM_KER_OS_NAME
fi
#FIXME: Print value of tunable kernel parameters, but dont define a env variable
echo -e "\nVarious tunable kernel parameters:"
echo "transparent_hugepage: "$(cat /sys/kernel/mm/transparent_hugepage/enabled)
echo "numa_balancing_scan_delay_ms: "$(cat /proc/sys/kernel/numa_balancing_scan_delay_ms)
echo "numa_balancing_scan_size_mb: "$(cat /proc/sys/kernel/numa_balancing_scan_size_mb)
echo "numa_balancing_scan_period_min_ms: "$(cat /proc/sys/kernel/numa_balancing_scan_period_min_ms)
echo "sched_tunable_scaling: "$(cat /proc/sys/kernel/sched_tunable_scaling)
echo "sched_rr_timeslice_ms: "$(cat /proc/sys/kernel/sched_rr_timeslice_ms)
echo "randomize_va_space: "$(cat /proc/sys/kernel/randomize_va_space)
echo "sched_min_granularity_ns: "$(cat /proc/sys/kernel/sched_min_granularity_ns)
echo "sched_wakeup_granularity_ns: "$(cat /proc/sys/kernel/sched_wakeup_granularity_ns)
echo "sched_migration_cost_ns: "$(cat /proc/sys/kernel/sched_migration_cost_ns)
echo "sched_nr_migrate: "$(cat /proc/sys/kernel/sched_nr_migrate)
echo "sched_rt_runtime_us: "$(cat /proc/sys/kernel/sched_rt_runtime_us)
echo "sched_latency_ns: "$(cat /proc/sys/kernel/sched_latency_ns)
echo "numa_balancing: "$(cat /proc/sys/kernel/numa_balancing)
echo "transparent_hugepage_defrag: "$(cat /sys/kernel/mm/transparent_hugepage/defrag)
echo "dirty_expire_centisecs: "$(cat /proc/sys/vm/dirty_expire_centisecs)
echo "dirty_writeback_centisecs: "$(cat /proc/sys/vm/dirty_writeback_centisecs)
echo "sched_cfs_bandwidth_slice_us: "$(cat /proc/sys/kernel/sched_cfs_bandwidth_slice_us)
if type -P cpupower >/dev/null 2>&1;
    then
        echo "cpupower frequency-info: "$(cpupower frequency-info)
    else
        echo "cpupower is not installed, cpupower frequency-info information not available"
    fi
if type -P tuned-adm >/dev/null 2>&1;
    then
        echo "tuned-adm profile: "$(tuned-adm profile)
    else
        echo "tuned-adm is not installed, tuned-adm profile information not available"
    fi

# Print BIOS details
export _SYSTEM_BIOS_VERSION=$(sudo dmidecode -s bios-version)
export _SYSTEM_BIOS_VENDOR=$(sudo dmidecode -s bios-vendor)
export _SYSTEM_BIOS_RELEASE_DATE=$(sudo dmidecode -s bios-release-date)
export _SYSTEM_BIOS_PROC_FAMILY=$(sudo dmidecode -s processor-family 2>&1 | head -n 1)
#export _SYSTEM_BIOS_PROC_VERSION=$(sudo dmidecode -s processor-version 2>&1 | head -n 1)

if [[ "$_HW_OS_KERNEL_BIOS_EXPORT_PRINT" = true ]]; then
    echo ""
    echo "BIOS Details:"
    echo "_SYSTEM_BIOS_VERSION: "$_SYSTEM_BIOS_VERSION
    echo "_SYSTEM_BIOS_VENDOR: "$_SYSTEM_BIOS_VENDOR
    echo "_SYSTEM_BIOS_RELEASE_DATE: "$_SYSTEM_BIOS_RELEASE_DATE
    echo "_SYSTEM_BIOS_PROC_FAMILY: "$_SYSTEM_BIOS_PROC_FAMILY
fi

