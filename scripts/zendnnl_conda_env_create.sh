#!/bin/bash
#*******************************************************************************
# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
#*******************************************************************************

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
		return
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

export PYTORCH_CONDA_ENV="zendnnltorch"
export PYTHON_VERSION=3.9

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
# conda init bash
# # source ~/.bashrc
# eval "$(conda shell.bash hook)"
# if [ $? -ne 0 ]; then
#     echo "(FAILED)"
#     echo "=> conda bash initialization failed."
#     return
# fi

# conda config --set auto_activate_base false
conda activate $PYTORCH_CONDA_ENV
if [ $? -ne 0 ]; then
    echo -n "Not found. Creating..."
    conda create -n $PYTORCH_CONDA_ENV python=$PYTHON_VERSION -y
    conda activate $PYTORCH_CONDA_ENV
    conda config --add channels conda-forge
    conda config --set channel_priority strict

    conda install -y cmake bazel make ninja doxygen openblas
    pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
    #pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    echo "(DONE)"
else
    echo "(FOUND)"
fi

conda deactivate

