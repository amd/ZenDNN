#*******************************************************************************
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license.
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

#Set BLIS PATH
BLIS_PATH:= ${ZENDNN_BLIS_PATH}

DEPEND_ON_CK ?= 0

ifeq ($(DEPEND_ON_CK), 1)
# Set Composable Kernel paths
	CK_PATH := ${ZENDNN_CK_PATH}
	CK_LIB_PATH := $(CK_PATH)/build/lib
	CK_DEFINES:= -DCK_NOGPU -DENABLE_CK
	CK_COMMON_FLAGS:= -Wno-attributes -Wno-ignored-attributes -Wno-write-strings
	CK_INCLUDES := -I$(CK_PATH) -I$(CK_PATH)/include -I$(CK_PATH)/library/include
	CPP_STD := c++17
else
	CPP_STD := c++14
endif

#Set LIBM PATH
ifeq "$(ZENDNN_ENABLE_LIBM)" "1"
	LIBM_PATH:= $(ZENDNN_LIBM_PATH)
	LIBM_INCLUDE_PATH:= -I$(LIBM_PATH)/include
	LIBM_LIB_PATH:= -L$(LIBM_PATH)/lib -lalm
	LIBM_ENABLE:= -DLIBM_ENABLE=1
else
	LIBM_PATH:=
	LIBM_INCLUDE_PATH:=
	LIBM_LIB_PATH:=
	LIBM_ENABLE:= -DLIBM_ENABLE=0
endif

#Compare if GCC version is 9 or above, so that we can use -march=znver2
GCCVERSIONGTEQ9 := $(shell expr `g++ -dumpversion | cut -f1 -d.` \>= 9)

#Select appropriate znver based on EPYC Model name
#Naming convention:
#model name:         AMD EPYC 7543 32-Core Processor
#Rome processors   : 7xx2
#Milan processors  : 7xx3

#Find the last digit of EPYC model number
EPYC_FAMILY_LAST_DIGIT := $(shell cat /proc/cpuinfo | grep 'model name' -m1 | awk '{print substr($$6, 4);}')
ZNVER=znver2 #Default value

ifeq "$(EPYC_FAMILY_LAST_DIGIT)" "3"
	ZNVER=znver3 #For Milan
endif

ifeq ($(RELEASE), 0)
ifeq ($(AOCC), 0)
	CXXFLAGS := -std=$(CPP_STD) -O0 -g -c -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -ggdb
	CXXFLAGSTEST := -std=$(CPP_STD) -O0 -g -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -ggdb
	CXXFLAGSGTEST := -std=$(CPP_STD) -O0 -g -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -ggdb
	CXXFLAGSBENCHTEST := -std=$(CPP_STD) -O0 -g -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -ggdb
	CXXFLAGSONEDNN := -std=$(CPP_STD) -O0 -g -fPIC -fopenmp
	COMMONFLAGS := -Werror -Wreturn-type -fconcepts -DZENDNN_X64=1 $(CK_COMMON_FLAGS)
	ifeq "$(GCCVERSIONGTEQ9)" "1"
		COMMONFLAGS += -march=znver2
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
	COMMONFLAGS := -Werror -Wreturn-type -fconcepts -DZENDNN_X64=1 $(CK_COMMON_FLAGS)
	ifeq "$(GCCVERSIONGTEQ9)" "1"
		COMMONFLAGS += -march=znver2
	else
		COMMONFLAGS += -march=znver1
	endif
else
	CXXFLAGS := -std=$(CPP_STD) -O3 -march=$(ZNVER) -c -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES) -DNDEBUG
	CXXFLAGSTEST := -std=$(CPP_STD) -O3 -march=$(ZNVER) -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES)
	CXXFLAGSGTEST := -std=$(CPP_STD) -O3 -march=$(ZNVER) -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES)
	CXXFLAGSBENCHTEST := -std=$(CPP_STD) -O3 -march=$(ZNVER) -fPIC -fopenmp -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 $(CK_DEFINES)
	CXXFLAGSONEDNN := -std=$(CPP_STD) -O3 -march=$(ZNVER) -fPIC -fopenmp
	COMMONFLAGS := -Wreturn-type -DZENDNN_X64=1
endif #AOCC
endif #RELEASE = 1


CXX_PREFIX ?= ccache
ifeq ($(AOCC), 0)
	CXX      := $(CXX_PREFIX) g++ $(LIBM_ENABLE)
else
	CXX      := $(CXX_PREFIX) clang++ $(LIBM_ENABLE)
endif

# https://github.com/mapbox/cpp/issues/37
# https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html

AR_COMMAND := ar rcs

PRODUCT  := libamdZenDNN.so
PRODUCT_ARCHIVE  := libamdZenDNN.a
CXXLINK  := -shared -fopenmp
LIBDIR   := lib
OBJDIR   := obj
OUTDIR   := _out
TESTDIR  := tests
ZENDNN_GIT_ROOT := $(shell pwd)

INCDIRS  := -Iinc -Isrc -Isrc/common -Isrc/cpu \
	-I$(BLIS_PATH)/include $(LIBM_INCLUDE_PATH) \
	$(CK_INCLUDES)

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
	src/cpu/x64/utils/*.cpp )
#$(info SRCS is $(SRCS))

OBJECT_FILES  := $(SRCS:%.cpp=$(OUTDIR)/$(OBJDIR)/%.o)
#$(info OBJECT_FILES is $(OBJECT_FILES))

ifeq ($(ARCHIVE), 1)
	all_target := create_dir build_so build_ar
else
	all_target := create_dir build_so
endif

all      : $(all_target)

build_so : $(EXECUTABLE_SO)
build_ar : $(EXECUTABLE_ARCHIVE)

#$@ => $(ZENDNN_GIT_ROOT)/$(OUTDIR)/$(LIBDIR)/$(PRODUCT) => $(ZENDNN_GIT_ROOT)/_out/lib/libamdZenDNN.so
#$^ => $(OBJS)
$(EXECUTABLE_SO): $(OBJECT_FILES)
	$(CXX) $(CXXLINK) -o $@ $^
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
	$(CXX) $(CXXFLAGS) $(COMMONFLAGS) $(INCDIRS) $< -o $@
	@# ^^^ Use $(CFLAGS), not $(LDFLAGS), when compiling.

clean:
	rm -fr $(OUTDIR)

create_dir:
	@mkdir -p $(OUTDIR)/$(LIBDIR)
	@mkdir -p $(OUTDIR)/$(TESTDIR)

ck_conv_test: $(OUTDIR)/$(LIBDIR)/$(PRODUCT)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_conv_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_conv_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH) \
		-L$(CK_LIB_PATH) -lck_cpu_instance -lhost_tensor

test: $(OUTDIR)/$(LIBDIR)/$(PRODUCT)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_conv_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_conv_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_inference_f32 $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_inference_f32.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_param $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_param.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_param_direct $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_param_direct.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_maxpool $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_maxpool.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/ref_avx_conv_param $(INCDIRS) \
		-Itests/api_tests tests/api_tests/ref_avx_conv_param.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_param_fusion $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_param_fusion.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_primitive_cache_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_primitive_cache_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_gelu_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_gelu_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_maxpool_blocked $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_maxpool_blocked.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/embedding_bag_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_embedding_bag_test.cpp -L_out/lib -lamdZenDNN \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)

test_archive: $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_conv_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_conv_test.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_inference_f32 $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_inference_f32.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_param $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_param.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_param_direct $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_param_direct.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_maxpool $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_maxpool.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/ref_avx_conv_param $(INCDIRS) \
		-Itests/api_tests tests/api_tests/ref_avx_conv_param.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_param_fusion $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_param_fusion.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_conv_primitive_cache_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_conv_primitive_cache_test.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_test.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_matmul_gelu_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_matmul_gelu_test.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/zendnn_avx_maxpool_blocked $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_avx_maxpool_blocked.cpp $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)
	$(CXX) $(CXXFLAGSTEST) $(COMMONFLAGS) -o $(OUTDIR)/$(TESTDIR)/embedding_bag_test $(INCDIRS) \
		-Itests/api_tests tests/api_tests/zendnn_embedding_bag_test.cpp  $(OUTDIR)/$(LIBDIR)/$(PRODUCT_ARCHIVE) \
		-L$(BLIS_PATH)/lib/ -lblis-mt $(LIBM_LIB_PATH)

.PHONY: all build_so test clean
