#*******************************************************************************
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license.
#*******************************************************************************

#===============================================================================
# Copyright 2016-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================


# propagate EXAMPLE specific flags
append(CMAKE_C_FLAGS "${CMAKE_EXAMPLE_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_EXAMPLE_CCXX_FLAGS}")

# propagate sanitizer flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")

# propagate nowarn flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

include_directories_with_host_compiler(${PROJECT_SOURCE_DIR}/include)
include_directories_with_host_compiler(${PROJECT_SOURCE_DIR}/examples)

append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_EXAMPLE_CXX_FLAGS}")
append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_CXX_NOWARN_FLAGS}")

file(GLOB_RECURSE sources *.cpp *.c)
file(GLOB_RECURSE headers *.hpp *.h)

# Skip TPP examples
foreach(f ${sources})
    get_filename_component(f_name ${f} NAME)
    if(NOT ZENDNN_ENABLE_TPP AND ${f_name} MATCHES "zendnn_tpp_integration_test.cpp")
        list(REMOVE_ITEM sources "${f}")
    endif()
endforeach()

# Skip CUSTOM OP examples
foreach(f ${sources})
    get_filename_component(f_name ${f} NAME)
    if(NOT ZENDNN_CUSTOM_OP_ENABLE AND ${f_name} MATCHES "^zendnn_grp")
        list(REMOVE_ITEM sources "${f}")
    endif()
endforeach()

# Skip SYCL, GPU and cross-engine examples
foreach(f ${sources})
    get_filename_component(f_name ${f} NAME)
    if(ZENDNN_GPU_RUNTIME STREQUAL "NONE" AND ${f_name} MATCHES "^(cross_engine|gpu)")
        list(REMOVE_ITEM sources "${f}")
    endif()
    if(NOT ZENDNN_WITH_SYCL AND ${f_name} MATCHES "^sycl")
        list(REMOVE_ITEM sources "${f}")
    endif()
endforeach()

# In case of SYCL, skip CPU examples that directly work with raw pointers
if(ZENDNN_CPU_SYCL)
    foreach(f ${sources})
        get_filename_component(fname ${f} NAME)
        if(${fname} MATCHES "cpu_")
            list(REMOVE_ITEM sources "${f}")
        endif()
    endforeach()
endif()

# Skip examples for CUDA since USM is a default model for the library which is
# not yet supported for Nvidia backend.
if(ZENDNN_SYCL_CUDA)
    foreach(f ${sources})
        list(REMOVE_ITEM sources "${f}")
    endforeach()
endif()

foreach(src ${sources})
    file(RELATIVE_PATH src_rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${src})
    string(REGEX REPLACE "[/_\\.]" "-" example_name ${src_rel_path})

    # Put hw-specific part of the name in front.
    # It is important for examples in subdirectories.
    foreach(pat "cpu-" "gpu-" "cross-engine-")
        string(REGEX REPLACE "^(.*)${pat}" "${pat}\\1"
            example_name ${example_name})
    endforeach()

    if(${example_name} MATCHES "(cross-engine|cpu|gpu)-")
        if(NOT ZENDNN_CPU_RUNTIME STREQUAL "NONE" OR ${example_name} MATCHES "gpu-")
            # Example name contains cross-engine, cpu or gpu
            if(NOT ${example_name} MATCHES ".*opencl" OR ZENDNN_GPU_RUNTIME STREQUAL "OCL")
                register_exe(${example_name} ${src} "test")
            endif()
        endif()
    else()
        set(cpu_rt_pattern "(SEQ|OMP|TBB|SYCL|DPCPP)")
        set(gpu_rt_pattern "(OCL|SYCL|DPCPP)")
        if(${example_name} MATCHES "sycl.*")
            set(cpu_rt_pattern "(SYCL|DPCPP)")
            set(gpu_rt_pattern "(SYCL|DPCPP)")
        endif()
        if(ZENDNN_CPU_RUNTIME MATCHES ${cpu_rt_pattern})
            # Adding test for CPU
            add_ZENDNN_test("cpu-${example_name}" "${example_name}" cpu)
            maybe_configure_windows_test("cpu-${example_name}" TEST)
        endif()
        if(ZENDNN_GPU_RUNTIME MATCHES ${gpu_rt_pattern})
            # Adding test for GPU
            add_ZENDNN_test("gpu-${example_name}" "${example_name}" gpu)
            maybe_configure_windows_test("gpu-${example_name}" TEST)
        endif()
        find_libm(LIBM)
        register_exe(${example_name} ${src} "" ${LIBM})
    endif()
endforeach()

if (ZENDNN_INSTALL_MODE STREQUAL "BUNDLE")
    configure_file(CMakeLists.txt.in CMakeLists.txt @ONLY)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeLists.txt
        ${sources} ${headers}
        DESTINATION examples)

    if(ZENDNN_WITH_SYCL)
        install(FILES
            ${PROJECT_SOURCE_DIR}/cmake/dpcpp_driver_check.cmake
            DESTINATION examples)
    else()
    # Skip matmul examples with SYCL
        install(DIRECTORY
            tutorials
            DESTINATION examples)
    endif()
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        install(FILES
            ${PROJECT_SOURCE_DIR}/cmake/template.vcxproj.user
            DESTINATION examples)
    endif()
endif()
