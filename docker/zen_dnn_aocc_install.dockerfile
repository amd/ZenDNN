# ZenDNN supported OS
FROM ubuntu:20.04

# Deps Directory
ENV ZEN_DNN_DEPS_ROOT=/opt/zen-dnn-deps
WORKDIR $ZEN_DNN_DEPS_ROOT

# set symbolic links to sh to use bash 
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Update Docker Image
RUN apt-get update -y

# install base dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ cmake pkg-config git sudo wget

# install ZEN DNN Deps - AOCC & AOCL
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install wget unzip python3-dev dmidecode && \
	wget https://developer.amd.com/wordpress/media/files/aocl-linux-aocc-3.0-6.tar.gz && \
	tar -xvf aocl-linux-aocc-3.0-6.tar.gz && cd aocl-linux-aocc-3.0-6/ && \
	tar -xvf aocl-blis-linux-aocc-3.0-6.tar.gz && cd ../ && \
	wget  https://developer.amd.com/wordpress/media/files/aocc-compiler-3.2.0.tar && \
	tar -xvf aocc-compiler-3.2.0.tar && cd aocc-compiler-3.2.0 && bash install.sh

# Install Zen DNN required Packages
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install numactl libnuma-dev hwloc
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install hwloc-nox ccache libopenblas-dev

# set environment variable
ENV ZENDNN_AOCC_COMP_PATH=$ZEN_DNN_DEPS_ROOT/aocc-compiler-3.2.0
ENV ZENDNN_BLIS_PATH=$ZEN_DNN_DEPS_ROOT/aocl-linux-aocc-3.0-6/amd-blis
ENV ZENDNN_LIBM_PATH=/usr/lib/x86_64-linux-gnu

# Working Directory
ENV ZEN_DNN_WORKING_ROOT=/workspace
WORKDIR $ZEN_DNN_WORKING_ROOT

# set OMP variables
RUN echo "export OMP_NUM_THREADS=$(grep -c ^processor /proc/cpuinfo)" >> ~/.profile
RUN echo "export GOMP_CPU_AFFINITY=\"0-$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')\"" >> ~/.profile

# set environment variable
ENV ZENDNN_GIT_ROOT=$ZEN_DNN_WORKING_ROOT/ZenDNN

# install Zen DNN
RUN DEBIAN_FRONTEND=noninteractive git clone https://github.com/amd/ZenDNN.git && cd ZenDNN && make clean && \
	source scripts/zendnn_aocc_build.sh

ENTRYPOINT source ~/.profile && /bin/bash