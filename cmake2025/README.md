/*******************************************************************************
* Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

# CMake build

ZenDNN can be built using CMake and the earlier Makefile based build will not be supported. Users are strongly encouraged to build using CMake.

CMake builds ZenDNN library and also its dependencies AMD-BLIS and FBGEMM

Source code is fetched from Git repositories below,

AMD-BLIS git repo,
https://github.com/amd/blis.git

FBGEMM git repo,
https://github.com/pytorch/FBGEMM.git

# Building on Linux
ZENDNN_BLIS_PATH environment variable refers to path of pre-built libraries or non existent directory for installing newly built libraries and is mandatory 

FBGEMM_INSTALL_PATH environment variable refers to path of pre-built libraries and is optional

in bash shell, example below, change to required paths for your host

```
export ZENDNN_BLIS_PATH=/scratch/zendnn_deps_install/AMDBLIS
export PYTHON_PATH=/scratch/miniforge3/bin/python
```

optional<br>

```
export FBGEMM_INSTALL_PATH=/scratch/zendnn_deps_install/FBGEMM
```

ZenDNN library build fetches the source code for AMD-BLIS and FBGEMM from git repositories using default Git Tags and are set in the CMakeLists.txt and those options can be overridden

provide build option in the command line, in case of alternate values

## common options
-D_GLIBCXX_USE_CXX11_ABI=0<br>
default: 0<br>
other possbile values: 0, 1

## AMD-BLIS options
-DAMDBLIS_TAG=AOCL-LPGEMM-012925<br>
A Git Tag to fetch the source code<br>
default: AOCL-LPGEMM-012925<br>
Users can provide any valid Git Tag

-DAMDBLIS_LOCAL_SOURCE=<local path of AMD-BLIS source><br>
instead of fetching source code from Git, alternately source code can referred from a local path, when this option is provided the option -DAMDBLIS_TAG is not considered.

Refer to AOCL-user-guide for more details<br>
## AOCL-user-guide<br>
https://docs.amd.com/go/en-US/57404-AOCL-user-guide

-DAMDBLIS_ENABLE_BLAS=ON<br>
default: ON<br>
possible values: ON, OFF

-DAMDBLIS_BLIS_CONFIG_FAMILY=amdzen<br>
default: amdzen<br>
for other possible values, Refer to AOCL-user-guide

-DAMDBLIS_ENABLE_THREADING<br>
default: openmp<br>
for other possible values, Refer to AOCL-user-guide

-DAMDBLIS_ENABLE_ADDON<br>
default: aocl_gemm<br>
for other possible values, Refer to AOCL-user-guide


## FBGEMM options

-DFBGEMM_TAG=v0.6.0<br>
A Git Tag to fetch the source code<br>
default: v0.6.0<br>
Users can provide any valid Git Tag


## ZENDNN options
-DLPGEMM_V5_0=1

-DFBGEMM_ENABLE=0<br>
possible values: 0, 1<br>
for 1, FBGEMM pre-built binary is checked and used if available, other wise will be built.

-DBLIS_API=0<br>
default: 0<br>
possbile values: 0, 1

-DBUILD_SHARED_LIBS=ON<br>
default: ON<br>
possbile values: ON, OFF<br>
for ON, means generate shared libraries<br>
for OFF, means generate static libraries<br>

-DCMAKE_BUILD_TYPE<br>
default: Release<br>
possible values: CMAKE supported values and Debug, Release, RelWithDebInfo are verified

# Build Examples:

## Example 1:
```
rm -rf build 
cmake -DLPGEMM_V5_0=1 -DFBGEMM_ENABLE=1 -DBLIS_API=1 -B build 
cd build 
cmake --build . --parallel $(nproc)
```

## Example 2:
```
rm -rf build 
cmake -DBUILD_SHARED_LIBS=ON -DAMDBLIS_LOCAL_SOURCE=/scratch/blis -DCMAKE_BUILD_TYPE=Debug -DAMDBLIS_ENABLE_BLAS=ON -D_GLIBCXX_USE_CXX11_ABI=0 -DLPGEMM_V5_0=1 -DFBGEMM_ENABLE=1 -DBLIS_API=1 -B build
cd build 
cmake --build . --parallel $(nproc)
```

# Standalone Tests:
```
cd <zendnn dir>
source scripts/zendnn_gcc_env_setup.sh
ZENDNN_LOG_OPTS=ALL:2  ./build/tests/zendnn_conv_test cpu
```
