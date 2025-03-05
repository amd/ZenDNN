#*******************************************************************************
# Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#******************************************************************************/

###########################################
# Makefile to compile amdZenDNN
#
##########################################
RELEASE ?= 1
AOCC ?= 1
# Set ARCHIVE to 1 to build libamdZenDNN.a
# make -j ARCHIVE=1
ARCHIVE ?= 0
BLIS_API ?= 0

# build for forced architecture. zero indicates arch is not forced
GXX_COMPILER = g++

# Set ZENDNN_STANDALONE_BUILD to 1 to build ZenDNN stanalone library
# BLIS include path and lib path is not same when BLIS is build from source vs
# BLIS official release is used
# Set LPGEMM to 1 to enable LPGEMM based Conv
# make -j LPGEMM=1
LPGEMM ?= 0
LPGEMM_V4_2 ?= 0

#Set BLIS PATH
ifeq "$(ZENDNN_STANDALONE_BUILD)" "1"
	BLIS_INC_PATH:=${ZENDNN_BLIS_PATH}/include/LP64
	BLIS_LIB_PATH:=${ZENDNN_BLIS_PATH}/lib/LP64
else
	BLIS_INC_PATH:=${ZENDNN_BLIS_PATH}/include
	BLIS_LIB_PATH:=${ZENDNN_BLIS_PATH}/lib
endif

DEPEND_ON_CK ?= 0

ifeq "$(LPGEMM)" "1"
       LPGEMM_ENABLE:= -DZENDNN_ENABLE_LPGEMM=1
endif

ifeq "$(LPGEMM_V4_2)" "1"
       LPGEMM_ENABLE:= -DZENDNN_ENABLE_LPGEMM=1 -DZENDNN_ENABLE_LPGEMM_V4_2=1
endif

ifeq "$(LPGEMM_V5_0)" "1"
       LPGEMM_ENABLE:= -DZENDNN_ENABLE_LPGEMM_V5_0=1
endif

ifeq ($(DEPEND_ON_CK), 1)
# Set Composable Kernel paths
	CK_PATH := ${ZENDNN_CK_PATH}
	CK_LINK_FLAGS := -L$(CK_PATH)/build/lib -lck_cpu_instance -lhost_tensor
	CK_DEFINES:= -DCK_NOGPU -DENABLE_CK
	CK_COMMON_FLAGS:= -Wno-attributes -Wno-ignored-attributes -Wno-write-strings
	CK_INCLUDES := -I$(CK_PATH) -I$(CK_PATH)/include -I$(CK_PATH)/library/include
endif

#Set uProf path
ifeq "$(ZENDNN_ENABLE_UPROF)" "1"
	UPROF_PATH:=$(UPROF_INSTALL_PATH)
	UPROF_INCLUDE_PATH:= -I$(UPROF_PATH)/include
	UPROF_LIB_PATH:= -L$(UPROF_PATH)/lib/x64
	UPROF_ENABLE:= -DUPROF_ENABLE=1
	UPROF_LINK:= -lAMDProfileController
	CXX_UPROF_LINK:= -Wl,--whole-archive $(UPROF_PATH)/lib/x64/libAMDProfileController.a -Wl,--no-whole-archive
else
	UPROF_PATH:=
	UPROF_INCLUDE_PATH:=
	UPROF_LIB_PATH:=
	UPROF_ENABLE:= -DUPROF_ENABLE=0
	CXX_UPROF_LINK :=
endif

#Set FBGEMM PATH
ifeq "$(FBGEMM_ENABLE)" "1"
	FBGEMM_PATH:= $(FBGEMM_INSTALL_PATH)
	FBGEMM_INCLUDE_PATH:= -I$(FBGEMM_PATH)/include
	FBGEMM_LIB_PATH:= -L$(FBGEMM_PATH)/build -lfbgemm
	USE_FBGEMM:= -DFBGEMM_ENABLE=1
else
	FBGEMM_PATH:=
	FBGEMM_INCLUDE_PATH:=
	FBGEMM_LIB_PATH:=
	USE_FBGEMM:= -DFBGEMM_ENABLE=0
endif

#Set LIBXSMM PATH(retained for future libxsmm integration)
ifeq "$(ZENDNN_ENABLE_LIBXSMM)" "1"
	LIBXSMM_PATH:= $(ZENDNN_LIBXSMM_PATH)
	LIBXSMM_INCLUDE_PATH:= -I$(LIBXSMM_PATH)/include
	LIBXSMM_LIB_PATH:= -L$(LIBXSMM_PATH)/lib -ldl -lxsmm -lxsmmnoblas -lxsmmext
else
	LIBXSMM_PATH:=
	LIBXSMM_INCLUDE_PATH:=
	LIBXSMM_LIB_PATH:=
endif

ifeq "$(ZENDNN_TF_USE_CUSTOM_BLIS)" "1"
	USE_CUSTOM_BLIS:= -DUSE_CUSTOM_BLIS=1
else
	USE_CUSTOM_BLIS:= -DUSE_CUSTOM_BLIS=0
endif
#Compare if GCC version is 9 or above, so that we can use -march=znver2
GCCVERSIONGTEQ12 := $(shell expr `g++ -dumpversion | cut -f1 -d.` \>= 12)

ifeq "$(GCCVERSIONGTEQ12)" "1"
	CPP_STD := c++17
	AVX512_BF16_EN=1
else
	CPP_STD := c++14
	AVX512_BF16_EN=0
endif
#Select appropriate znver based on EPYC Model name
#Naming convention:
#model name        : AMD EPYC 7543 32-Core Processor
#Rome processors   : 7xx2
#Milan processors  : 7xx3

ZNVER=znver2 #Default value

ifeq ($(RELEASE), 0)
ifeq ($(AOCC), 0)
	CXXFLAGS := -std=$(CPP_STD) -O0 -g -c -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -ggdb
	CXXFLAGSTEST := -std=$(CPP_STD) -O0 -g -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -ggdb
	CXXFLAGSGTEST := -std=$(CPP_STD) -O0 -g -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -ggdb
	CXXFLAGSBENCHTEST := -std=$(CPP_STD) -O0 -g -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -ggdb
	CXXFLAGSONEDNN := -std=$(CPP_STD) -O0 -g -fPIC -fopenmp
	COMMONFLAGS := -Wno-error -Wreturn-type -fconcepts -DZENDNN_X64=1 $(CK_COMMON_FLAGS)
	ifeq "$(GCCVERSIONGTEQ12)" "1"
		COMMONFLAGS += -march=$(ZNVER)
	else
		COMMONFLAGS += -march=znver1
	endif
else
	CXXFLAGS := -std=$(CPP_STD) -O0 -march=$(ZNVER) -g -c -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -ggdb
	CXXFLAGSTEST := -std=$(CPP_STD) -O0 -march=$(ZNVER) -g -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -ggdb
	CXXFLAGSGTEST := -std=$(CPP_STD) -O0 -march=$(ZNVER) -g -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -ggdb
	CXXFLAGSBENCHTEST := -std=$(CPP_STD) -O0 -march=$(ZNVER) -g -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -ggdb
	CXXFLAGSONEDNN := -std=$(CPP_STD) -O0 -march=$(ZNVER) -g -fPIC -fopenmp
	COMMONFLAGS := -Wreturn-type -DZENDNN_X64=1
endif #AOCC
else #RELEASE = 1
ifeq ($(AOCC), 0)
	CXXFLAGS := -std=$(CPP_STD) -O3 -c -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -DNDEBUG
	CXXFLAGSTEST := -std=$(CPP_STD) -O3 -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES)
	CXXFLAGSGTEST := -std=$(CPP_STD) -O3 -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES)
	CXXFLAGSBENCHTEST := -std=$(CPP_STD) -O3 -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES)
	CXXFLAGSONEDNN := -std=$(CPP_STD) -O3 -fPIC -fopenmp
	ifeq ($(LPGEMM), 1)
	     COMMONFLAGS := -Wreturn-type -fconcepts -DZENDNN_X64=1 $(CK_COMMON_FLAGS) -lstdc++ -mssse3 -Wno-deprecated $(LPGEMM_ENABLE) -DAVX512_BF16_EN=$(AVX512_BF16_EN)
	else
	     COMMONFLAGS := -Wno-error -Wreturn-type -fconcepts -DZENDNN_X64=1 $(CK_COMMON_FLAGS)
	endif
	ifeq "$(GCCVERSIONGTEQ12)" "1"
		COMMONFLAGS += -march=$(ZNVER)
	else
		COMMONFLAGS += -march=znver1
	endif
else
	CXXFLAGS := -std=$(CPP_STD) -O3 -march=$(ZNVER) -c -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -DNDEBUG
	CXXFLAGSTEST := -std=$(CPP_STD) -O3 -march=$(ZNVER) -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES)
	CXXFLAGSGTEST := -std=$(CPP_STD) -O3 -march=$(ZNVER) -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES)
	CXXFLAGSBENCHTEST := -std=$(CPP_STD) -O3 -march=$(ZNVER) -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES)
	CXXFLAGSONEDNN := -std=$(CPP_STD) -O3 -march=$(ZNVER) -fPIC -fopenmp
	COMMONFLAGS := -Wreturn-type -DZENDNN_X64=1 $(LPGEMM_ENABLE) -DAVX512_BF16_EN=$(AVX512_BF16_EN)
endif #AOCC
endif #RELEASE = 1

ifeq ($(BLIS_API), 0)
	COMMONFLAGS += -UZENDNN_USE_AOCL_BLIS_API
else
	COMMONFLAGS += -DZENDNN_USE_AOCL_BLIS_API
endif

COMMONFLAGS += -Wno-format-zero-length -Wno-format-truncation
COMMONFLAGS += -Wno-unused-result -Wno-stringop-overflow
COMMONFLAGS += -Wno-format -Wno-narrowing

CXX_PREFIX ?= ccache
ifeq ($(AOCC), 0)
	CXX := $(CXX_PREFIX) $(GXX_COMPILER) $(USE_CUSTOM_BLIS) $(USE_FBGEMM) $(UPROF_ENABLE)
else
	CXX := $(CXX_PREFIX) clang++ $(USE_CUSTOM_BLIS) $(UPROF_ENABLE)
endif

# https://github.com/mapbox/cpp/issues/37
# https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html

AR_COMMAND := ar rcs

PRODUCT  := libamdZenDNN.so
PRODUCT_ARCHIVE  := libamdZenDNN.a
CXXLINK  := -shared -fopenmp
LIBDIR	 := lib
OBJDIR	 := obj
OUTDIR	 := _out
TESTDIR  := tests
ZENDNN_GIT_ROOT := $(shell pwd)

INCDIRS  := -Iinc -Isrc -Isrc/common -Isrc/cpu -Isrc/tpp \
	-I$(BLIS_INC_PATH) $(FBGEMM_INCLUDE_PATH) \
	$(UPROF_INCLUDE_PATH) $(CK_INCLUDES) \
	$(CK_INCLUDES) $(LIBXSMM_INCLUDE_PATH)

EXECUTABLE_SO := $(ZENDNN_GIT_ROOT)/$(OUTDIR)/$(LIBDIR)/$(PRODUCT)
EXECUTABLE_ARCHIVE := $(ZENDNN_GIT_ROOT)/$(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE)

SRCS := $(wildcard src/common/*.cpp \
	src/cpu/*.cpp \
	src/cpu/gemm/*.cpp \
	src/cpu/gemm/f32/*.cpp \
	src/cpu/gemm/s8x8s32/*.cpp \
	src/cpu/matmul/*.cpp \
	src/cpu/reorder/*.cpp \
	src/cpu/rnn/*.cpp \
	src/cpu/x64/*.cpp \
	src/cpu/x64/brgemm/*.cpp \
	src/cpu/x64/gemm/*.cpp \
	src/cpu/x64/gemm/amx/*.cpp \
	src/cpu/x64/gemm/bf16/*.cpp \
	src/cpu/x64/gemm/f32/*.cpp \
	src/cpu/x64/gemm/s8x8s32/*.cpp \
	src/cpu/x64/injectors/*.cpp \
	src/cpu/x64/lrn/*.cpp \
	src/cpu/x64/matmul/*.cpp \
	src/cpu/x64/prelu/*.cpp \
	src/cpu/x64/rnn/*.cpp \
	src/cpu/x64/shuffle/*.cpp \
	src/cpu/x64/utils/*.cpp)

#$(info SRCS is $(SRCS))

OBJECT_FILES  := $(SRCS:%.cpp=$(OUTDIR)/$(OBJDIR)/%.o)
#$(info OBJECT_FILES is $(OBJECT_FILES))

ifeq ($(ARCHIVE), 1)
	all_target := create_dir build_so build_ar
else
	all_target := create_dir build_so
endif

all		 : $(all_target)

build_so : $(EXECUTABLE_SO)
build_ar : $(EXECUTABLE_ARCHIVE)

#$@ => $(ZENDNN_GIT_ROOT)/$(OUTDIR)/$(LIBDIR)/$(PRODUCT) => $(ZENDNN_GIT_ROOT)/_out/lib/libamdZenDNN.so
#$^ => $(OBJS)
$(EXECUTABLE_SO): $(OBJECT_FILES)
	$(CXX) $(CXXLINK) $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) $(CXX_UPROF_LINK) -o $@ $^
	@# ^^^ http://www.gnu.org/software/make/manual/make.html#Automatic-Variables
	@echo "Build successful (shared)!"

#$@ => $(ZENDNN_GIT_ROOT)/$(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) => $(ZENDNN_GIT_ROOT)/_out/lib/libamdZenDNN.a
#$^ => $(OBJS)
$(EXECUTABLE_ARCHIVE): $(OBJECT_FILES)
	$(AR_COMMAND) $@ $^
	@# ^^^ http://www.gnu.org/software/make/manual/make.html#Automatic-Variables
	@echo "Build successful (archive)!"

# http://www.gnu.org/software/make/manual/make.html#Static-Pattern
$(OBJECT_FILES): $(OUTDIR)/$(OBJDIR)/%.o: %.cpp
	@# @echo Compiling $<
	@# ^^^ Your terminology is weird: you "compile a .cpp file" to create a .o file.
	@mkdir -p $(@D)
	@# ^^^ http://www.gnu.org/software/make/manual/make.html#index-_0024_0028_0040D_0029
	$(CXX) $(CXXFLAGS) $(COMMONFLAGS) $(INCDIRS) $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) $(UPROF_LIB_PATH) $(UPROF_LINK) $< -o $@
	@# ^^^ Use $(CFLAGS), not $(LDFLAGS), when compiling.

clean:
	rm -fr $(OUTDIR)

create_dir:
	@mkdir -p $(OUTDIR)/$(LIBDIR)
	@mkdir -p $(OUTDIR)/$(TESTDIR)

test: $(OUTDIR)/$(LIBDIR)/$(PRODUCT)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_attention_multihead_f32 $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_multihead_attention_f32.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_attention_multihead_bf16 $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_multihead_attention_bf16.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_conv_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_conv_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_sdpa_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_sdpa_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_inference_f32 $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_inference_f32.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_param $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_param.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_param_direct $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_param_direct.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_maxpool $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_maxpool.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/ref_avx_conv_param $(INCDIRS) \
		-Itests/api_tests tests/api_tests/ref_avx_conv_param.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_param_fusion $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_param_fusion.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_primitive_cache_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_primitive_cache_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_int $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_int8_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_inplace $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_inplace_custom_op_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_block_size $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_block_size.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_generic_int8 $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_generic_int8_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_int4 $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_int4_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_grp_matmul_test $(INCDIRS) \
                -Itests/api_tests tests/api_tests/zendnn_grp_matmul_test.cpp -L_out/lib -lamdZenDNN \
                -L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
                $(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmulFusions_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmulFusions_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_gelu_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_gelu_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_weight_cache_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_weight_cache_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_bf16_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_bf16_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_maxpool_blocked $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_maxpool_blocked.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/embedding_bag_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_embedding_bag_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
		$(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/embedding_bag_benchmark $(INCDIRS) \
                -Itests/api_tests tests/api_tests/zendnn_embedding_bag_benchmark.cpp -L_out/lib -lamdZenDNN \
                -L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
                $(CK_LINK_FLAGS)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/grp_embedding_bag_test $(INCDIRS) \
                -Itests/api_tests tests/api_tests/zendnn_grp_embedding_bag_test.cpp -L_out/lib -lamdZenDNN \
                -L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
                $(CK_LINK_FLAGS)
ifeq "$(FBGEMM_ENABLE)" "1"
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/grp_embedding_bag_test_int4 $(INCDIRS) \
                -Itests/api_tests tests/api_tests/zendnn_grp_embedding_bag_test_int4.cpp -L_out/lib -lamdZenDNN \
                -L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
                $(CK_LINK_FLAGS)
endif
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/grp_embedding_mlp_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_grp_embedding_mlp_test.cpp -L_out/lib -lamdZenDNN \
                -L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH) \
                $(CK_LINK_FLAGS)

test_archive: $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_conv_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_conv_test.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_inference_f32 $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_inference_f32.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_param $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_param.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_param_direct $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_param_direct.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_maxpool $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_maxpool.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/ref_avx_conv_param $(INCDIRS) \
		-Itests/api_tests tests/api_tests/ref_avx_conv_param.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_param_fusion $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_param_fusion.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_primitive_cache_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_primitive_cache_test.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_test.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_grp_matmul_test $(INCDIRS) \
                -Itests/api_tests tests/api_tests/zendnn_grp_matmul_test.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
                -L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmulFusions_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmulFusions_test.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_gelu_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_gelu_test.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_weight_cache_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_weight_cache_test.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_bf16_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_bf16_test.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_maxpool_blocked $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_maxpool_blocked.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/embedding_bag_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_embedding_bag_test.cpp  $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_int $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_int8_test.cpp  $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_inplace $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_inplace_custom_op_test.cpp  $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_block_size $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_block_size.cpp  $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_generic_int8 $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_generic_int8_test.cpp  $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_int4 $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_int4_test.cpp  $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/grp_embedding_bag_test $(INCDIRS) \
                -Itests/api_tests tests/api_tests/zendnn_grp_embedding_bag_test.cpp  $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
                -L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/grp_embedding_mlp_test $(INCDIRS) \
                -Itests/api_tests tests/api_tests/zendnn_grp_embedding_mlp_test.cpp  $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
                -L$(BLIS_LIB_PATH) -lblis-mt $(FBGEMM_LIB_PATH) $(LIBXSMM_LIB_PATH)

.PHONY: all build_so test clean
