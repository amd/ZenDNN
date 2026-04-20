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
            --local-amdblis-dir )
                ZENDNNL_LOCAL_AMDBLIS=1
                ZENDNNL_LOCAL_AMDBLIS_DIR=$2
                shift
                shift
                ;;
            --local-aocldlp-dir )
                ZENDNNL_LOCAL_AOCLDLP=1
                ZENDNNL_LOCAL_AOCLDLP_DIR=$2
                shift
                shift
                ;;
            --local-onednn-dir )
                ZENDNNL_LOCAL_ONEDNN=1
                ZENDNNL_LOCAL_ONEDNN_DIR=$2
                shift
                shift
                ;;
            --local-libxsmm-dir )
                ZENDNNL_LOCAL_LIBXSMM=1
                ZENDNNL_LOCAL_LIBXSMM_DIR=$2
                shift
                shift
                ;;
            --local-parlooper-dir )
                ZENDNNL_LOCAL_PARLOOPER=1
                ZENDNNL_LOCAL_PARLOOPER_DIR=$2
                shift
                shift
                ;;
            --local-fbgemm-dir )
                ZENDNNL_LOCAL_FBGEMM=1
                ZENDNNL_LOCAL_FBGEMM_DIR=$2
                shift
                shift
                ;;
            --inject-aocldlp )
                ZENDNNL_INJECT_AOCLDLP=$2
                shift
                shift
                ;;
            --inject-amdblis )
                ZENDNNL_INJECT_AMDBLIS=$2
                shift
                shift
                ;;
            --inject-onednn )
                ZENDNNL_INJECT_ONEDNN=$2
                shift
                shift
                ;;
            --inject-libxsmm )
                ZENDNNL_INJECT_LIBXSMM=$2
                shift
                shift
                ;;
            --inject-parlooper )
                ZENDNNL_INJECT_PARLOOPER=$2
                shift
                shift
                ;;
            --inject-fbgemm )
                ZENDNNL_INJECT_FBGEMM=$2
                shift
                shift
                ;;
            --debug )
                ZENDNNL_DEBUG_BUILD=1
                shift
                ;;
            --shared )
                ZENDNNL_SHARED_LIB=1
                shift
                ;;
            --asan )
                ZENDNNL_ASAN=1
                shift
                ;;
            --coverage )
                ZENDNNL_COVERAGE=1
                shift
                ;;
            --install-prefix )
                ZENDNNL_INSTALL_PREFIX_OPT=$2
                shift
                shift
                ;;
            --cc )
                ZENDNNL_CC=$2
                shift
                shift
                ;;
            --cxx )
                ZENDNNL_CXX=$2
                shift
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
                echo " local dependency options (builds from local source) :"
                echo " --local-amdblis    : use local amdblis from dependencies/amdblis."
                echo " --local-aocldlp    : use local aocldlp from dependencies/aocldlp."
                echo " --local-aoclutils  : use local aoclutils from dependencies/aoclutils."
                echo " --local-json       : use local json from dependencies/json."
                echo " --local-onednn     : use local onednn from dependencies/onednn."
                echo " --local-libxsmm    : use local libxsmm from dependencies/libxsmm."
                echo " --local-parlooper  : use local parlooper from dependencies/parlooper."
                echo " --local-fbgemm     : use local fbgemm from dependencies/fbgemm."
                echo
                echo " local dependency with custom path (builds from specified source dir) :"
                echo " --local-amdblis-dir <path>    : use amdblis source from <path>."
                echo " --local-aocldlp-dir <path>    : use aocldlp source from <path>."
                echo " --local-onednn-dir <path>     : use onednn source from <path>."
                echo " --local-libxsmm-dir <path>    : use libxsmm source from <path>."
                echo " --local-parlooper-dir <path>  : use parlooper source from <path>."
                echo " --local-fbgemm-dir <path>     : use fbgemm source from <path>."
                echo
                echo " dependency injection (use pre-built install, no compilation) :"
                echo " --inject-aocldlp <path>    : inject pre-built aocldlp from <path>."
                echo " --inject-amdblis <path>    : inject pre-built amdblis from <path>."
                echo " --inject-onednn <path>     : inject pre-built onednn from <path>."
                echo " --inject-libxsmm <path>    : inject pre-built libxsmm from <path>."
                echo " --inject-parlooper <path>  : inject pre-built parlooper from <path>."
                echo " --inject-fbgemm <path>     : inject pre-built fbgemm from <path>."
                echo
                echo " build options :"
                echo " --nproc <N>        : number of processes for parallel build (default: 1)."
                echo " --debug            : build in debug mode (default: release)."
                echo " --shared           : build shared library (.so) in addition to static."
                echo " --asan             : enable AddressSanitizer."
                echo " --coverage         : enable code coverage."
                echo " --install-prefix <path> : set custom install prefix."
                echo " --cc <path>        : set C compiler (e.g. /usr/bin/gcc-13)."
                echo " --cxx <path>       : set C++ compiler (e.g. /usr/bin/g++-13)."
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
                echo "   # build with pre-built aocl-dlp and libxsmm"
                echo "   source zendnnl_build.sh --zendnnl --inject-aocldlp /opt/aocl-dlp --inject-libxsmm /opt/libxsmm"
                echo
                echo "   # build from local source at custom path"
                echo "   source zendnnl_build.sh --zendnnl --local-onednn-dir /home/user/onednn"
                echo
                echo "   # rebuild without re-downloading dependencies"
                echo "   source zendnnl_build.sh --no-deps --all"
                echo
                echo "   # debug build with ASAN"
                echo "   source zendnnl_build.sh --zendnnl --debug --asan"
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
ZENDNNL_LOCAL_AMDBLIS_DIR=""
ZENDNNL_LOCAL_AOCLDLP_DIR=""
ZENDNNL_LOCAL_ONEDNN_DIR=""
ZENDNNL_LOCAL_LIBXSMM_DIR=""
ZENDNNL_LOCAL_PARLOOPER_DIR=""
ZENDNNL_LOCAL_FBGEMM_DIR=""
ZENDNNL_INJECT_AOCLDLP=""
ZENDNNL_INJECT_AMDBLIS=""
ZENDNNL_INJECT_ONEDNN=""
ZENDNNL_INJECT_LIBXSMM=""
ZENDNNL_INJECT_PARLOOPER=""
ZENDNNL_INJECT_FBGEMM=""
ZENDNNL_DEBUG_BUILD=0
ZENDNNL_SHARED_LIB=0
ZENDNNL_ASAN=0
ZENDNNL_COVERAGE=0
ZENDNNL_INSTALL_PREFIX_OPT=""
ZENDNNL_CC=""
ZENDNNL_CXX=""

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

# Helper function to validate custom local source directory
check_local_dir() {
    local dep_name=$1
    local local_dir=$2
    if [ ! -d "${local_dir}" ]; then
        echo "error: local source directory for ${dep_name} not found at ${local_dir}"
        return 1
    fi
    return 0
}

# Helper function to validate inject path
check_inject_path() {
    local dep_name=$1
    local inject_path=$2
    if [ ! -d "${inject_path}" ]; then
        echo "error: inject path for ${dep_name} not found at ${inject_path}"
        return 1
    fi
    if [ ! -d "${inject_path}/lib" ] && [ ! -d "${inject_path}/lib64" ]; then
        echo "warning: ${inject_path} does not contain a lib/ directory. injection may fail."
    fi
    if [ ! -d "${inject_path}/include" ]; then
        echo "warning: ${inject_path} does not contain an include/ directory. injection may fail."
    fi
    return 0
}

# Validate local dependencies (only when no custom dir is given)
if [ ${ZENDNNL_LOCAL_AMDBLIS} -eq 1 ] && [ -z "${ZENDNNL_LOCAL_AMDBLIS_DIR}" ]; then
    if ! check_local_dep "amdblis"; then return 1; fi
fi
if [ ${ZENDNNL_LOCAL_AOCLDLP} -eq 1 ] && [ -z "${ZENDNNL_LOCAL_AOCLDLP_DIR}" ]; then
    if ! check_local_dep "aocldlp"; then return 1; fi
fi
if [ ${ZENDNNL_LOCAL_AOCLUTILS} -eq 1 ]; then
    if ! check_local_dep "aoclutils"; then return 1; fi
fi
if [ ${ZENDNNL_LOCAL_JSON} -eq 1 ]; then
    if ! check_local_dep "json"; then return 1; fi
fi
if [ ${ZENDNNL_LOCAL_ONEDNN} -eq 1 ] && [ -z "${ZENDNNL_LOCAL_ONEDNN_DIR}" ]; then
    if ! check_local_dep "onednn"; then return 1; fi
fi
if [ ${ZENDNNL_LOCAL_LIBXSMM} -eq 1 ] && [ -z "${ZENDNNL_LOCAL_LIBXSMM_DIR}" ]; then
    if ! check_local_dep "libxsmm"; then return 1; fi
fi
if [ ${ZENDNNL_LOCAL_PARLOOPER} -eq 1 ] && [ -z "${ZENDNNL_LOCAL_PARLOOPER_DIR}" ]; then
    if ! check_local_dep "parlooper"; then return 1; fi
fi
if [ ${ZENDNNL_LOCAL_FBGEMM} -eq 1 ] && [ -z "${ZENDNNL_LOCAL_FBGEMM_DIR}" ]; then
    if ! check_local_dep "fbgemm"; then return 1; fi
fi

# Validate custom local source directories
if [ -n "${ZENDNNL_LOCAL_AMDBLIS_DIR}" ]; then
    if ! check_local_dir "amdblis" "${ZENDNNL_LOCAL_AMDBLIS_DIR}"; then return 1; fi
fi
if [ -n "${ZENDNNL_LOCAL_AOCLDLP_DIR}" ]; then
    if ! check_local_dir "aocldlp" "${ZENDNNL_LOCAL_AOCLDLP_DIR}"; then return 1; fi
fi
if [ -n "${ZENDNNL_LOCAL_ONEDNN_DIR}" ]; then
    if ! check_local_dir "onednn" "${ZENDNNL_LOCAL_ONEDNN_DIR}"; then return 1; fi
fi
if [ -n "${ZENDNNL_LOCAL_LIBXSMM_DIR}" ]; then
    if ! check_local_dir "libxsmm" "${ZENDNNL_LOCAL_LIBXSMM_DIR}"; then return 1; fi
fi
if [ -n "${ZENDNNL_LOCAL_PARLOOPER_DIR}" ]; then
    if ! check_local_dir "parlooper" "${ZENDNNL_LOCAL_PARLOOPER_DIR}"; then return 1; fi
fi
if [ -n "${ZENDNNL_LOCAL_FBGEMM_DIR}" ]; then
    if ! check_local_dir "fbgemm" "${ZENDNNL_LOCAL_FBGEMM_DIR}"; then return 1; fi
fi

# Validate inject paths
if [ -n "${ZENDNNL_INJECT_AOCLDLP}" ]; then
    if ! check_inject_path "aocldlp" "${ZENDNNL_INJECT_AOCLDLP}"; then return 1; fi
fi
if [ -n "${ZENDNNL_INJECT_AMDBLIS}" ]; then
    if ! check_inject_path "amdblis" "${ZENDNNL_INJECT_AMDBLIS}"; then return 1; fi
fi
if [ -n "${ZENDNNL_INJECT_ONEDNN}" ]; then
    if ! check_inject_path "onednn" "${ZENDNNL_INJECT_ONEDNN}"; then return 1; fi
fi
if [ -n "${ZENDNNL_INJECT_LIBXSMM}" ]; then
    if ! check_inject_path "libxsmm" "${ZENDNNL_INJECT_LIBXSMM}"; then return 1; fi
fi
if [ -n "${ZENDNNL_INJECT_PARLOOPER}" ]; then
    if ! check_inject_path "parlooper" "${ZENDNNL_INJECT_PARLOOPER}"; then return 1; fi
fi
if [ -n "${ZENDNNL_INJECT_FBGEMM}" ]; then
    if ! check_inject_path "fbgemm" "${ZENDNNL_INJECT_FBGEMM}"; then return 1; fi
fi

# create build folder
cd ${parent_dir}
if [ ! -d "build" ];then
    echo "creating ${parent_dir}/build directory..."
    mkdir -p build
fi
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

    # build type
    if [ ${ZENDNNL_DEBUG_BUILD} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_BUILD_TYPE=Debug"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_BUILD_TYPE=Release"
    fi

    # custom compilers
    if [ -n "${ZENDNNL_CC}" ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_C_COMPILER=${ZENDNNL_CC}"
    fi
    if [ -n "${ZENDNNL_CXX}" ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_CXX_COMPILER=${ZENDNNL_CXX}"
    fi

    # custom install prefix
    if [ -n "${ZENDNNL_INSTALL_PREFIX_OPT}" ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_INSTALL_PREFIX=${ZENDNNL_INSTALL_PREFIX_OPT}"
    fi

    # shared library
    if [ ${ZENDNNL_SHARED_LIB} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LIB_BUILD_SHARED=ON"
    fi

    # ASAN
    if [ ${ZENDNNL_ASAN} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_BUILD_ASAN=ON"
    fi

    # code coverage
    if [ ${ZENDNNL_COVERAGE} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_CODE_COVERAGE=ON"
    fi

    # dependencies
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

    # local dependency options
    if [ ${ZENDNNL_LOCAL_AMDBLIS} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_AOCLDLP=OFF -DZENDNNL_DEPENDS_AMDBLIS=ON -DZENDNNL_LOCAL_AMDBLIS=ON"
        if [ -n "${ZENDNNL_LOCAL_AMDBLIS_DIR}" ];then
            CMAKE_OPTIONS="${CMAKE_OPTIONS} -DAMDBLIS_ROOT_DIR=${ZENDNNL_LOCAL_AMDBLIS_DIR}"
        fi
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_AMDBLIS=OFF"
    fi

    if [ ${ZENDNNL_LOCAL_AOCLDLP} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_AOCLDLP=ON"
        if [ -n "${ZENDNNL_LOCAL_AOCLDLP_DIR}" ];then
            CMAKE_OPTIONS="${CMAKE_OPTIONS} -DAOCLDLP_ROOT_DIR=${ZENDNNL_LOCAL_AOCLDLP_DIR}"
        fi
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
        if [ -n "${ZENDNNL_LOCAL_ONEDNN_DIR}" ];then
            CMAKE_OPTIONS="${CMAKE_OPTIONS} -DONEDNN_ROOT_DIR=${ZENDNNL_LOCAL_ONEDNN_DIR}"
        fi
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_ONEDNN=OFF"
    fi

    if [ ${ZENDNNL_LOCAL_LIBXSMM} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_LIBXSMM=ON"
        if [ -n "${ZENDNNL_LOCAL_LIBXSMM_DIR}" ];then
            CMAKE_OPTIONS="${CMAKE_OPTIONS} -DLIBXSMM_ROOT_DIR=${ZENDNNL_LOCAL_LIBXSMM_DIR}"
        fi
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_LIBXSMM=OFF"
    fi

    if [ ${ZENDNNL_LOCAL_PARLOOPER} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_PARLOOPER=ON -DZENDNNL_LOCAL_PARLOOPER=ON"
        if [ -n "${ZENDNNL_LOCAL_PARLOOPER_DIR}" ];then
            CMAKE_OPTIONS="${CMAKE_OPTIONS} -DPARLOOPER_ROOT_DIR=${ZENDNNL_LOCAL_PARLOOPER_DIR}"
        fi
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_PARLOOPER=OFF"
    fi

    if [ ${ZENDNNL_LOCAL_FBGEMM} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_FBGEMM=ON"
        if [ -n "${ZENDNNL_LOCAL_FBGEMM_DIR}" ];then
            CMAKE_OPTIONS="${CMAKE_OPTIONS} -DFBGEMM_ROOT_DIR=${ZENDNNL_LOCAL_FBGEMM_DIR}"
        fi
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_FBGEMM=OFF"
    fi

    # dependency injection options
    if [ -n "${ZENDNNL_INJECT_AOCLDLP}" ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_AOCLDLP=ON -DZENDNNL_AOCLDLP_INJECT_DIR=${ZENDNNL_INJECT_AOCLDLP}"
    fi
    if [ -n "${ZENDNNL_INJECT_AMDBLIS}" ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_AOCLDLP=OFF -DZENDNNL_DEPENDS_AMDBLIS=ON -DZENDNNL_AMDBLIS_INJECT_DIR=${ZENDNNL_INJECT_AMDBLIS}"
    fi
    if [ -n "${ZENDNNL_INJECT_ONEDNN}" ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_ONEDNN=ON -DZENDNNL_ONEDNN_INJECT_DIR=${ZENDNNL_INJECT_ONEDNN}"
    fi
    if [ -n "${ZENDNNL_INJECT_LIBXSMM}" ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_LIBXSMM=ON -DZENDNNL_LIBXSMM_INJECT_DIR=${ZENDNNL_INJECT_LIBXSMM}"
    fi
    if [ -n "${ZENDNNL_INJECT_PARLOOPER}" ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_PARLOOPER=ON -DZENDNNL_PARLOOPER_INJECT_DIR=${ZENDNNL_INJECT_PARLOOPER}"
    fi
    if [ -n "${ZENDNNL_INJECT_FBGEMM}" ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_FBGEMM=ON -DZENDNNL_FBGEMM_INJECT_DIR=${ZENDNNL_INJECT_FBGEMM}"
    fi

    # build component options
    if [[ ${ZENDNNL_GTEST} -eq 1 || ${ZENDNNL_ALL} -eq 1 ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_BUILD_GTEST=ON"
    fi

    if [[ ${ZENDNNL_BENCHDNN} -eq 1 || ${ZENDNNL_ALL} -eq 1 ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_BUILD_BENCHDNN=ON"
    fi

    if [[ ${ZENDNNL_DOXYGEN} -eq 1 || ${ZENDNNL_ALL} -eq 1 ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_BUILD_DOXYGEN=ON"
    fi

    if [[ ${ZENDNNL_EXAMPLES} -eq 1 || ${ZENDNNL_ALL} -eq 1 ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_BUILD_EXAMPLES=ON"
    fi

    if [ ${ZENDNNL_ALL} -eq 1 ];then
        echo "cmake ${CMAKE_OPTIONS} .."
        cmake ${CMAKE_OPTIONS} ..

        if [ $? -eq 0 ]; then
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
                echo "cmake --build . --target ${TARGET_OPTIONS} -j${ZENDNNL_NPROC}"
                cmake --build . --target ${TARGET_OPTIONS} -j${ZENDNNL_NPROC}
            fi
        else
            echo "no targets given... nothing to do."
        fi
    fi
fi

cd ${curr_dir}
