#!/bin/bash
# *******************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#Sanity Check
#0. if Conda activated or not
#1. if gcov installed or not
#2. if gcov==g++ versin
#3. if lcov installed or not
function sanity_check() {

  #0. if conda set
  if [[ -n "$CONDA_PREFIX" ]]; then
    echo "Conda environment is activated: $CONDA_DEFAULT_ENV"
  else
    echo "No Conda environment is currently activated."
    exit 1;
  fi

  # Check if gcov is installed
  if ! command -v gcov &> /dev/null; then
    echo "gcov is not installed."
    exit 1
  fi
 
  # Get versions
  gcc_version=$(gcc --version | head -n1 | awk '{print $NF}')
  gcpp_version=$(g++ --version | head -n1 | awk '{print $NF}')
  gcov_version=$(gcov --version | head -n1 | awk '{print $NF}')

  echo "gcc version: $gcc_version"
  echo "g++ version: $gcpp_version"
  echo "gcov version: $gcov_version"
 
  # Compare versions
  if [[ "$gcc_version" == "$gcpp_version"  && "$gcc_version" == "$gcov_version"  ]]; then
    echo "gcc, g++ and gcov versions match."
  else
    echo "Version mismatch: "
    [[ "$gcc_version" != "$gcpp_version" ]] && echo " - gcc and g++ differ"
    [[ "$gcc_version" != "$gcov_version" ]] && echo " - gcc and gcov differ"
    exit 1
  fi

  # check for lcov
  if ! command -v lcov &> /dev/null; then
    echo "lcov is not installed."
    exit 1
  fi
}

#Set Path
function setup() {
  curr_dir="$(pwd)"
  parent_dir="$(dirname "$curr_dir")"
  last_dir="$(basename $curr_dir)"
  if [ ${last_dir} != "scripts" ];then
    echo "error: <${last_dir}> does not seem to be <scripts> folder."
    return 1;
  fi
}

#Build with Code Coverage Support
function code_coverage_build() {
  cd $parent_dir
  if [ ! -d "build" ];then
    echo "creating ${parent_dir}/build directory..."
    mkdir -p build
  fi
  cd build
  cmake .. -DZENDNNL_BUILD_GTEST=ON -DZENDNNL_CODE_COVERAGE=ON
  # This will generate gcno files.
  cmake --build . --target=all
}

#Generate Document
function generate_report() {
  #This will generate gcda files.
  ./install/gtests/gtests

  OUTPUT_INFO="coverage.info"
  FILTERED_INFO="coverage_filtered.info"
  HTML_DIR="coverage_html"

  echo "Capturing coverage data..."
  lcov --capture --directory . --output-file "$OUTPUT_INFO"
 
  echo "Filtering out system and test files..."
  lcov --remove "$OUTPUT_INFO" '/usr/*' '*/gtest/*' --output-file "$FILTERED_INFO"
 
  echo "Generating HTML report..."
  genhtml "$FILTERED_INFO" --output-directory "$HTML_DIR"

  echo
  echo "Code Coverage report for ZENDNN(L) is generated."
  echo "Open $HTML_DIR/index.html manually in your browser."
}

#init()
function init() {
  sanity_check
  setup
  code_coverage_build
  generate_report
}

init
