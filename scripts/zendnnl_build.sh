#!/bin/bash
# *******************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
	    --enable-onednn )
                ZENDNNL_DEPENDS_ONEDNN=1
                shift
		;;
	    --enable-aocldlp )
                ZENDNNL_DEPENDS_AOCLDLP=1
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
            --nproc )
                ZENDNNL_NPROC=$2
                shift
                shift
                ;;
            --help )
                echo " usage   : zendnnl-build <options>"
                echo
                echo " options :"
                echo " --all              : build and install all targets."
                echo " --clean            : clean all targets."
                echo " --clean-all        : clean dependencies and build folders."
                echo " --zendnnl          : build and install zendnnl lib."
                echo " --zendnnl-gtest    : build and install zendnnl gtest."
                echo " --examples         : build and install examples."
                echo " --benchdnn         : build and install benchdnn."
                echo " --doxygen          : build and install doxygen docs."
                echo " --no-deps          : don't rebuild (or clean) dependencies."
                echo " --enable-onednn    : enable onednn."
                echo " --enable-aocldlp   : enable aocldlp."
                echo " --local-amdblis    : use local amdblis."
                echo " --local-aocldlp    : use local aocldlp."
                echo " --local-aoclutils  : use local aoclutils."
                echo " --local-json       : use local json."
                echo " --local-onednn     : use local onednn."
                echo " --nproc            : number of processes for parallel build."
                echo
                echo " examples :"
                echo " build all targets including dependencies"
                echo " source zendnnl_build.sh --all"
                echo
                echo " build all targets if dependencies are already built"
                echo " (will fail if dependencies are not built by previous build)"
                echo " source zendnnl_build.sh --no-deps --all"
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
ZENDNNL=0
ZENDNNL_GTEST=0
ZENDNNL_EXAMPLES=0
ZENDNNL_BENCHDNN=0
ZENDNNL_DOXYGEN=0
ZENDNNL_NODEPS=0
ZENDNNL_DEPENDS_ONEDNN=0
ZENDNNL_DEPENDS_AOCLDLP=0
ZENDNNL_LOCAL_AMDBLIS=0
ZENDNNL_LOCAL_AOCLDLP=0
ZENDNNL_LOCAL_AOCLUTILS=0
ZENDNNL_LOCAL_JSON=0
ZENDNNL_LOCAL_ONEDNN=0
ZENDNNL_DEBUG_BUILD=0
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
# create build folder
# echo "switching to <${parent_dir}>."
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

    if [ ${ZENDNNL_DEPENDS_ONEDNN} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_ONEDNN=ON"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_ONEDNN=OFF"
    fi

    if [ ${ZENDNNL_DEPENDS_AOCLDLP} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_AMDBLIS=OFF"
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_DEPENDS_AOCLDLP=ON"
    fi

    if [ ${ZENDNNL_LOCAL_AMDBLIS} -eq 1 ];then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_LOCAL_AMDBLIS=ON"
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

    if [ ${ZENDNNL_ALL} -eq 1 ];then
        echo "cmake ${CMAKE_OPTIONS} .."
        cmake ${CMAKE_OPTIONS} ..

        if [ $? -eq 0 ]; then
            #cmake --build . --target clean
            echo "cmake --build . --target all -j${ZENDNNL_NPROC}"
            cmake --build . --target all
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
                cmake --build . --target ${TARGET_OPTIONS}
            fi
        else
            echo "no targets given... nothing to do."
        fi
    fi
fi
cd ${curr_dir}
