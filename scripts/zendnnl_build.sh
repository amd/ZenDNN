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
	    --zendnnl-nogtest )
		ZENDNNL_NOGTEST=1
		shift
		shift
		;;
	    --zendnnl )
		ZENDNNL=1
		shift
		;;
	    --examples )
		ZENDNNL_EXAMPLES=1
		shift
		;;
	    --doxygen )
		ZENDNNL_DOXYGEN=1
		shift
		;;
	    --all )
		ZENDNNL_ALL=1
		shift
		;;
	    --clean )
		ZENDNNL_CLEAN=1
		shift
		;;
	    --no-deps )
                ZENDNNL_NODEPS=1
                shift
		;;
            --help )
                echo " usage   : zendnnl-build <options>"
                echo
                echo " options :"
                echo " --all      : build and install all targets."
                echo " --clean    : clean all targets."
                echo " --zendnnl  : build and install zendnnl lib."
                echo " --examples : build and install examples."
                echo " --doxygen  : build and install doxygen docs."
                echo " --no-deps  : don't rebuild (or clean) dependencies."
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
ZENDNNL_NOGTEST=0
ZENDNNL=0
ZENDNNL_EXAMPLES=0
ZENDNNL_DOXYGEN=0
ZENDNNL_ALL=0
ZENDNNL_CLEAN=0
ZENDNNL_NODEPS=0

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


# go to build folder
echo "switching to ${parent_dir}/build ..."
cd build

# configure and build
CMAKE_OPTIONS=""
TARGET_OPTIONS=""

if [ ${ZENDNNL_NODEPS} -eq 1 ];then
    CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_BUILD_DEPS=OFF"
else
    CMAKE_OPTIONS="${CMAKE_OPTIONS} -DZENDNNL_BUILD_DEPS=ON"
fi

if [ ${ZENDNNL_ALL} -eq 1 ];then

    cmake ${CMAKE_OPTIONS} ..
    #cmake --build . --target clean
    cmake --build . --target all

elif [ ${ZENDNNL_CLEAN} -eq 1 ];then

    cmake ${CMAKE_OPTIONS} ..
    cmake --build . --target clean

else
    if [ ${ZENDNNL} -eq 1 ];then
        TARGET_OPTIONS="${TARGET_OPTIONS} zendnnl"
    fi

    if [ ${ZENDNNL_EXAMPLES} -eq 1 ];then
        TARGET_OPTIONS="${TARGET_OPTIONS} zendnnl-examples"
    fi

    if [ ${ZENDNNL_DOXYGEN} -eq 1 ];then
        TARGET_OPTIONS="${TARGET_OPTIONS} zendnnl-doxygen"
    fi

    if [[ ! -z ${TARGET_OPTIONS} ]];then
        echo "building targets ${TARGET_OPTIONS}"
        cmake ${CMAKE_OPTIONS} ..
        #cmake --build . --target clean
        cmake --build . --target ${TARGET_OPTIONS}
    else
        echo "no targets given... nothing to do."
    fi
fi


cd ${curr_dir}
