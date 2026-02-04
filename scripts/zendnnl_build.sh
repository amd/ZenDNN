#!/bin/bash
# *******************************************************************************
# * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#------------------------------------------------------------------------------
# functions
# argument parsing
function parse_args() {
    while [[ $# -gt 0 ]]
    do
        key="$1"
        case $key in
            --all )
                ZENDNNL_ALL=1
                shift
                ;;
            --clean )
                ZENDNNL_CLEAN=1
                shift
                ;;
            --clean-all )
                ZENDNNL_CLEAN_ALL=1
                shift
                ;;
            --zendnnl )
                ZENDNNL=1
                shift
                ;;
            --zendnnl-gtest )
                ZENDNNL_GTEST=1
                shift
                ;;
            --examples )
                ZENDNNL_EXAMPLES=1
                shift
                ;;
            --benchdnn )
                ZENDNNL_BENCHDNN=1
                shift
                ;;
            --doxygen )
                ZENDNNL_DOXYGEN=1
                shift
                ;;
            --no-deps )
                ZENDNNL_NODEPS=1
                shift
                ;;
            --enable-parlooper )
                ZENDNNL_DEPENDS_PARLOOPER=1
                shift
                ;;
            --enable-amdblis )
                ZENDNNL_DEPENDS_AMDBLIS=1
                shift
                ;;
            --local-amdblis )
                ZENDNNL_LOCAL_AMDBLIS=1
                shift
                ;;
            --local-aocldlp )
                ZENDNNL_LOCAL_AOCLDLP=1
                shift
                ;;
            --local-aoclutils )
                ZENDNNL_LOCAL_AOCLUTILS=1
                shift
                ;;
            --local-json )
                ZENDNNL_LOCAL_JSON=1
                shift
                ;;
            --local-onednn )
                ZENDNNL_LOCAL_ONEDNN=1
                shift
                ;;
            --local-libxsmm )
                ZENDNNL_LOCAL_LIBXSMM=1
                shift
                ;;
            --local-parlooper )
                ZENDNNL_LOCAL_PARLOOPER=1
                shift
                ;;
            --local-fbgemm )
                ZENDNNL_LOCAL_FBGEMM=1
                shift
                ;;
            --nproc )
                ZENDNNL_NPROC=$2
                shift
                shift
                ;;
            --help )
                echo " usage   : source zendnnl_build.sh <options>"
                echo
                echo " IMPORTANT: use --zendnnl to build the library along with other targets."
                echo "            Components like --zendnnl-gtest, --examples, --benchdnn depend on"
                echo "            the zendnnl library. Use --all to build all targets."
                echo
                echo " build targets :"
                echo " --all              : build all targets."
                echo " --zendnnl          : build zendnnl library (required for other components)."
                echo " --zendnnl-gtest    : build zendnnl gtests (requires --zendnnl)."
                echo " --examples         : build examples (requires --zendnnl)."
                echo " --benchdnn         : build benchdnn (requires --zendnnl)."
                echo " --doxygen          : build doxygen docs (standalone, no library required)."
                echo
                echo " clean options :"
                echo " --clean            : clean all targets."
                echo " --clean-all        : clean dependencies and build folders."
                echo
                echo " dependency options :"
                echo " --no-deps          : don't rebuild (or clean) dependencies."
                echo " --enable-parlooper : enable parlooper."
                echo " --enable-amdblis   : enable amdblis (disables aocldlp which is default)."
                echo
                echo " local dependency options (requires source in dependencies/<name>/) :"
                echo " --local-amdblis    : use local amdblis."
                echo " --local-aocldlp    : use local aocldlp."
                echo " --local-aoclutils  : use local aoclutils."
                echo " --local-json       : use local json."
                echo " --local-onednn     : use local onednn."
                echo " --local-libxsmm    : use local libxsmm."
                echo " --local-parlooper  : use local parlooper."
                echo " --local-fbgemm     : use local fbgemm."
                echo
                echo " build options :"
                echo " --nproc <N>        : number of processes for parallel build (default: 1)."
                echo
                echo " examples :"
                echo "   # build all targets including dependencies"
                echo "   source zendnnl_build.sh --all"
                echo
                echo "   # build all targets with parallel jobs"
                echo "   source zendnnl_build.sh --all --nproc 8"
                echo
                echo "   # build library only"
                echo "   source zendnnl_build.sh --zendnnl"
                echo
                echo "   # build library + gtests"
                echo "   source zendnnl_build.sh --zendnnl --zendnnl-gtest"
                echo
                echo "   # build library + gtests + examples"
                echo "   source zendnnl_build.sh --zendnnl --zendnnl-gtest --examples"
                echo
                echo "   # rebuild without re-downloading dependencies"
                echo "   source zendnnl_build.sh --no-deps --all"
                echo
                echo "   # use local onednn source"
                echo "   source zendnnl_build.sh --zendnnl --local-onednn"
                echo
                return 1
                ;;
            * )
                echo "unknown command line option $1"
                return 1
        esac
    done
    return 0
}

# parse arguments
ZENDNNL_ALL=0
ZENDNNL_CLEAN=0
ZENDNNL_CLEAN_ALL=0

# build target options
ZENDNNL=0
ZENDNNL_GTEST=0
ZENDNNL_EXAMPLES=0
ZENDNNL_BENCHDNN=0
ZENDNNL_DOXYGEN=0

# configure options
ZENDNNL_NODEPS=0
ZENDNNL_DEPENDS_PARLOOPER=0
ZENDNNL_DEPENDS_AMDBLIS=0
ZENDNNL_LOCAL_AMDBLIS=0
ZENDNNL_LOCAL_AOCLDLP=0
ZENDNNL_LOCAL_AOCLUTILS=0
ZENDNNL_LOCAL_JSON=0
ZENDNNL_LOCAL_ONEDNN=0
ZENDNNL_LOCAL_LIBXSMM=0
ZENDNNL_LOCAL_PARLOOPER=0
ZENDNNL_LOCAL_FBGEMM=0
ZENDNNL_DEBUG_BUILD=0

# build options
ZENDNNL_NPROC=1

if ! parse_args $@;
then
   return 1
fi

# sanity check
curr_dir="$(pwd)"
parent_dir="$(dirname "$curr_dir")"
last_dir="$(basename $curr_dir)"
if [ ${last_dir} != "scripts" ];then
    echo "error: <${last_dir}> does not seem to be <scripts> folder."
    return 1;
fi

# Helper function to check if local dependency exists
check_local_dep() {
    local dep_name=$1
    local dep_dir="${parent_dir}/dependencies/${dep_name}"
    if [ ! -d "${dep_dir}" ]; then
        echo "error: local ${dep_name} directory not found at ${dep_dir}"
        echo "       please clone/copy ${dep_name} to ${dep_dir} before using --local-${dep_name}"
        return 1
    fi
    return 0
}

# Validate local dependencies
if [ ${ZENDNNL_LOCAL_AMDBLIS} -eq 1 ]; then
    if ! check_local_dep "amdblis"; then return 1; fi
fi
if [ ${ZENDNNL_LOCAL_AOCLDLP} -eq 1 ]; then
    if ! check_local_dep "aocldlp"; then return 1; fi
fi
if [ ${ZENDNNL_LOCAL_AOCLUTILS} -eq 1 ]; then
    if ! check_local_dep "aoclutils"; then return 1; fi
fi
if [ ${ZENDNNL_LOCAL_JSON} -eq 1 ]; then
    if ! check_local_dep "json"; then return 1; fi
fi
if [ ${ZENDNNL_LOCAL_ONEDNN} -eq 1 ]; then
    if ! check_local_dep "onednn"; then return 1; fi
fi
if [ ${ZENDNNL_LOCAL_LIBXSMM} -eq 1 ]; then
    if ! check_local_dep "libxsmm"; then return 1; fi
fi
if [ ${ZENDNNL_LOCAL_PARLOOPER} -eq 1 ]; then
    if ! check_local_dep "parlooper"; then return 1; fi
fi
if [ ${ZENDNNL_LOCAL_FBGEMM} -eq 1 ]; then
    if ! check_local_dep "fbgemm"; then return 1; fi
fi

# create build folder
cd ${parent_dir}
if [ ! -d "build" ];then
    echo "creating ${parent_dir}/build directory..."
    mkdir -p build
# else
#     echo "<build> directory exists."
fi
# if [ ! -z "$(ls -A "./build")" ];then
#     echo "<build> is not empty. please empty it manually for fresh build."
# fi
if [ ${ZENDNNL_CLEAN_ALL} -eq 1 ];then
    echo "cleaning ${parent_dir}/dependencies..."
    cd ${parent_dir}/dependencies
    if [ $? -eq 0 ];then
        rm -rf *
    else
        echo "${parent_dir}/dependencies does not exist."
    fi
    echo "cleaning ${parent_dir}/build..."
    cd ${parent_dir}/build
    if [ $? -eq 0 ];then
        rm -rf *
    else
        echo "${parent_dir}/build does not exist."
    fi
    echo "cleaned dependencies and build folder. please rerun the script."
else
    # go to build folder
    echo "switching to ${parent_dir}/build ..."
    cd build

    # configure and build
    CMAKE_OPTIONS=""
    TARGET_OPTIONS=""

    if [ ${ZENDNNL_DEBUG_BUILD} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_BUILD_TYPE=Debug"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_BUILD_TYPE=Release"
    fi

    if [ ${ZENDNNL_NODEPS} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_BUILD_DEPS=OFF"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_BUILD_DEPS=ON"
    fi

    if [[ ${ZENDNNL_DEPENDS_PARLOOPER} -eq 1 || ${ZENDNNL_ALL} -eq 1 ]];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_PARLOOPER=ON"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_PARLOOPER=OFF"
    fi

    if [ ${ZENDNNL_DEPENDS_AMDBLIS} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_AOCLDLP=OFF"
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_AMDBLIS=ON"
    fi

    if [ ${ZENDNNL_LOCAL_AMDBLIS} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_AOCLDLP=OFF -DZENDNNL_DEPENDS_AMDBLIS=ON -DZENDNNL_LOCAL_AMDBLIS=ON"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_AMDBLIS=OFF"
    fi

    if [ ${ZENDNNL_LOCAL_AOCLDLP} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_AOCLDLP=ON"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_AOCLDLP=OFF"
    fi

    if [ ${ZENDNNL_LOCAL_AOCLUTILS} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_AOCLUTILS=ON"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_AOCLUTILS=OFF"
    fi

    if [ ${ZENDNNL_LOCAL_JSON} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_JSON=ON"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_JSON=OFF"
    fi

    if [ ${ZENDNNL_LOCAL_ONEDNN} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_ONEDNN=ON"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_ONEDNN=OFF"
    fi

    if [ ${ZENDNNL_LOCAL_LIBXSMM} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_LIBXSMM=ON"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_LIBXSMM=OFF"
    fi

    if [ ${ZENDNNL_LOCAL_PARLOOPER} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_PARLOOPER=ON -DZENDNNL_LOCAL_PARLOOPER=ON"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_PARLOOPER=OFF"
    fi

    if [ ${ZENDNNL_LOCAL_FBGEMM} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_FBGEMM=ON"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_FBGEMM=OFF"
    fi

    if [[ ${ZENDNNL_GTEST} -eq 1 || ${ZENDNNL_ALL} -eq 1 ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_BUILD_GTEST=ON"
    fi

    if [[ ${ZENDNNL_BENCHDNN} -eq 1 || ${ZENDNNL_ALL} -eq 1 ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_BUILD_BENCHDNN=ON"
    fi

    if [[ ${ZENDNNL_DOXYGEN} -eq 1 || ${ZENDNNL_ALL} -eq 1 ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_BUILD_DOXYGEN=ON"
    fi

    if [ ${ZENDNNL_ALL} -eq 1 ];then
        echo "cmake ${CMAKE_OPTIONS} .."
        cmake ${CMAKE_OPTIONS} ..

        if [ $? -eq 0 ]; then
            #cmake --build . --target clean
            echo "cmake --build . --target all -j${ZENDNNL_NPROC}"
            cmake --build . --target all -j${ZENDNNL_NPROC}
        fi
    elif [ ${ZENDNNL_CLEAN} -eq 1 ];then
        cmake ${CMAKE_OPTIONS} ..
        if [ $? -eq 0 ]; then
            echo "cmake --build . --target clean"
            cmake --build . --target clean
        fi
    else
        if [ ${ZENDNNL} -eq 1 ];then
            TARGET_OPTIONS="${TARGET_OPTIONS} zendnnl"
        fi
        if [ ${ZENDNNL_EXAMPLES} -eq 1 ];then
            TARGET_OPTIONS="${TARGET_OPTIONS} zendnnl-examples"
        fi
        if [ ${ZENDNNL_GTEST} -eq 1 ];then
            TARGET_OPTIONS="${TARGET_OPTIONS} zendnnl-gtest"
        fi
        if [ ${ZENDNNL_BENCHDNN} -eq 1 ];then
            TARGET_OPTIONS="${TARGET_OPTIONS} zendnnl-benchdnn"
        fi
        if [ ${ZENDNNL_DOXYGEN} -eq 1 ];then
            TARGET_OPTIONS="${TARGET_OPTIONS} zendnnl-doxygen"
        fi
        if [[ ! -z ${TARGET_OPTIONS} ]];then
            echo "building targets ${TARGET_OPTIONS}"
            cmake ${CMAKE_OPTIONS} ..
            if [ $? -eq 0 ]; then
                #cmake --build . --target clean
                echo "cmake --build . --target ${TARGET_OPTIONS} -j${ZENDNNL_NPROC}"
                cmake --build . --target ${TARGET_OPTIONS} -j${ZENDNNL_NPROC}
            fi
        else
            echo "no targets given... nothing to do."
        fi
    fi
fi

cd ${curr_dir}
