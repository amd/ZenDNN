#!/bin/bash

sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
#sudo apt install gcc-12 g++-12
sudo apt install gcc-13 g++-13
