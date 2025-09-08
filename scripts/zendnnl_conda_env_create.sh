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
	    -env | --env )
		PYTORCH_CONDA_ENV=$2
		shift
		shift
		;;
	    -d | --delete )
		DEL_ENV=1
		shift
		;;
	    -h | --help )
		echo "conda_set_env <options>"
		echo "-env [conda env to setup]"
		echo "-d (to remove existing env)"
		;;
	    * )
		echo "unknown command line option $1"
		return 0
	esac
    done

    return 0
}

# ask user
function ask_user() {
    read -p "$1 ([Y]ES or (n)o):"
    case $(echo $REPLY) in
	Y|YES) echo "yes" ;;
	*) echo "no" ;;
    esac
}

#------------------------------------------------------------------------------
# defaults

WORK_DIR=$(pwd)
DEL_ENV=0

if ! parse_args $@;
then
    return
fi

#------------------------------------------------------------------------------
# sanity check

export PYTORCH_CONDA_ENV="zendnnl_build"
export PYTHON_VERSION=3.10

echo -n "Checking if PYTORCH_CONDA_ENV is defined..."
if [ ! -z "$PYTORCH_CONDA_ENV" ];
then
    echo "(DONE)"
else
    echo "(FAILED)"
    echo "=> Please set PYTORCH_CONDA_ENV to proceed."
    return
fi

echo -n "Checking PYTHON_VERSION..."
if [ ! -z "$PYTHON_VERSION" ];
then
    echo "(DONE)"
else
    echo "(FAILED)"
    echo "=> Please provide PYTHON_VERSION to proceed."
    return
fi

# remove if requested for removal

if [ $DEL_ENV -eq 1 ];
then
    if [ "yes" == $(ask_user "Remove $PYTORCH_CONDA_ENV ") ];
    then
	conda remove --name $PYTORCH_CONDA_ENV --all
    fi
    return
fi

echo -n "Checking for conda virtual env $PYTORCH_CONDA_ENV..."
#source ~/anaconda3/etc/profile.d/conda.sh
conda init bash
# # source ~/.bashrc
eval "$(conda shell.bash hook)"
if [ $? -ne 0 ]; then
    echo "(FAILED)"
    echo "=> conda bash initialization failed."
    return
fi

# conda config --set auto_activate_base false
conda activate $PYTORCH_CONDA_ENV
if [ $? -ne 0 ]; then
    echo -n "Not found. Creating..."
    conda create -n $PYTORCH_CONDA_ENV python=$PYTHON_VERSION -y
    conda activate $PYTORCH_CONDA_ENV
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    conda install -y cmake bazel make ninja doxygen openblas clang  clangxx  llvmdev llvm-openmp
    #echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
	  #echo 'export CPATH=$CONDA_PREFIX/include' >> ~/.bashrc
    #echo 'export LIBRARY_PATH=$CONDA_PREFIX/lib' >> ~/.bashrc
    source ~/.bashrc
    #pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
    #pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    echo "(DONE)"
else
    echo "(FOUND)"
fi

conda deactivate

