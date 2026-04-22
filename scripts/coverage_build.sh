#!/bin/bash
# *******************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Run tests with different MATMUL configurations and collect coverage data
function run_and_generate_coverage_report() {
  OUTPUT_INFO="coverage.info"
  FILTERED_INFO="coverage_filtered.info"
  HTML_DIR="coverage_html"
  
  # ZENDNNL_MATMUL_ALGO values for lowoha=false: 1, 2, 4, 5
  ALGO_VALUES_LOWOHA_FALSE=(1 2 4 5)
  
  # ZENDNNL_MATMUL_ALGO values for lowoha=true: auto, 0, 1, 2, 3, 4, 5, 6
  ALGO_VALUES_LOWOHA_TRUE=("auto" 0 1 2 3 4 5 6)
  
  # ============================================================================
  # ENABLE ALL LOGGERS AND PROFILERS FOR MAXIMUM COVERAGE
  # ============================================================================
  # These environment variables enable logging code paths that would otherwise
  # be skipped, ensuring coverage of logger/profiler source code.
  # See docs/logging.md for detailed information about log levels.
  # ============================================================================
  export ZENDNNL_ENABLE_PROFILER=1         # Enable profiler code paths
  export ZENDNNL_PROFILE_LOG_LEVEL=4       # Verbose profiler output
  export ZENDNNL_COMMON_LOG_LEVEL=4        # Common logging enabled
  export ZENDNNL_API_LOG_LEVEL=4           # API logging enabled
  export ZENDNNL_TEST_LOG_LEVEL=4          # Test logging enabled
  
  echo "=========================================="
  echo "Logging/Profiler environment variables set:"
  echo "  ZENDNNL_ENABLE_PROFILER=1"
  echo "  ZENDNNL_PROFILE_LOG_LEVEL=4"
  echo "  ZENDNNL_COMMON_LOG_LEVEL=4"
  echo "  ZENDNNL_API_LOG_LEVEL=4"
  echo "  ZENDNNL_TEST_LOG_LEVEL=4"
  echo "=========================================="
  
  # ============================================================================
  # Run tests with lowoha=false
  # ============================================================================
  for algo in "${ALGO_VALUES_LOWOHA_FALSE[@]}"; do
    echo "=========================================="
    echo "Running tests with ZENDNNL_MATMUL_ALGO=$algo (lowoha=false) ..."
    echo "=========================================="
    export ZENDNNL_MATMUL_ALGO=$algo
    # Use --ai_test_mode coverage for strategic minimal tests with maximum code coverage
    # Alternative modes: pre-sub (minimal), post-sub (comprehensive), nightly (10x tests)
    ./install/gtests/gtests --ai_test_mode coverage --lowoha false --gtest_filter="AITests/*:-AITests/TestBatch*"
    
    # Capture coverage for this run
    lcov --capture --directory . --output-file "coverage_lowoha_false_algo_${algo}.info"
  done
  
  # ============================================================================
  # Run tests with lowoha=true
  # ============================================================================
  for algo in "${ALGO_VALUES_LOWOHA_TRUE[@]}"; do
    echo "=========================================="
    echo "Running tests with ZENDNNL_MATMUL_ALGO=$algo (lowoha=true) ..."
    echo "=========================================="
    export ZENDNNL_MATMUL_ALGO=$algo
    # Use --ai_test_mode coverage for strategic minimal tests with maximum code coverage
    # Alternative modes: pre-sub (minimal), post-sub (comprehensive), nightly (10x tests)
    ./install/gtests/gtests --ai_test_mode coverage --lowoha true --gtest_filter="AITests/*:-AITests/TestBatch*"
    
    # Capture coverage for this run
    lcov --capture --directory . --output-file "coverage_lowoha_true_algo_${algo}.info"
  done
  
  # ============================================================================
  # Run tests with logging DISABLED to cover else-branches
  # ============================================================================
  # This ensures both if(logging_enabled) and else branches get coverage
  # ============================================================================
  echo "=========================================="
  echo "Running tests with logging DISABLED for else-branch coverage..."
  echo "=========================================="
  unset ZENDNNL_ENABLE_PROFILER
  unset ZENDNNL_PROFILE_LOG_LEVEL
  unset ZENDNNL_COMMON_LOG_LEVEL
  unset ZENDNNL_API_LOG_LEVEL
  unset ZENDNNL_TEST_LOG_LEVEL
  
  export ZENDNNL_MATMUL_ALGO=auto
  ./install/gtests/gtests --ai_test_mode coverage --lowoha true --gtest_filter="AITests/*:-AITests/TestBatch*"
  
  # Capture coverage for this run
  lcov --capture --directory . --output-file "coverage_logging_disabled.info"
  
  # Build the lcov command dynamically based on both ALGO_VALUES arrays
  TRACEFILE_ARGS=""
  for algo in "${ALGO_VALUES_LOWOHA_FALSE[@]}"; do
    TRACEFILE_ARGS="$TRACEFILE_ARGS --add-tracefile coverage_lowoha_false_algo_${algo}.info"
  done
  for algo in "${ALGO_VALUES_LOWOHA_TRUE[@]}"; do
    TRACEFILE_ARGS="$TRACEFILE_ARGS --add-tracefile coverage_lowoha_true_algo_${algo}.info"
  done
  
  # Add logging-disabled coverage file
  TRACEFILE_ARGS="$TRACEFILE_ARGS --add-tracefile coverage_logging_disabled.info"
  
  # Merge all coverage files
  echo "=========================================="
  echo "Merging coverage data from all runs..."
  echo "=========================================="
  lcov $TRACEFILE_ARGS --output-file "$OUTPUT_INFO"
  
  # ============================================================================
  # FILTERING CONFIGURATION
  # ============================================================================
  # You can use POSITIVE filtering (include only), NEGATIVE filtering (exclude),
  # or BOTH (extract first, then remove from the extracted set)
  # ============================================================================
  
  # ----------------------------------------------------------------------------
  # POSITIVE FILTER: Specify directories/files to INCLUDE in coverage report
  # ----------------------------------------------------------------------------
  # Uncomment to use positive filtering (extracts ONLY the specified paths)
  # To add more paths, add them to the array:
  # Example: INCLUDE_PATHS=("*/zendnnl/src/*" "*/benchdnn/*" "*/examples/*")
  #
  # Note: Patterns should use wildcards (*) for lcov matching
  # ----------------------------------------------------------------------------
  INCLUDE_PATHS=(
    "*/zendnnl/*"
  )
  
  # ----------------------------------------------------------------------------
  # NEGATIVE FILTER: Patterns to EXCLUDE from coverage report
  # ----------------------------------------------------------------------------
  # Current exclusions: system files, gtest framework, test files, and build artifacts
  #
  # To add more exclusions, add them to the array:
  # Common patterns to exclude:
  # - */benchdnn/*          (benchmark code)
  # - */examples/*          (example code)
  # - */external/*          (external dependencies)
  # - */third_party/*       (third-party dependencies)
  # - */dependencies/*      (dependency directory)
  # - */CMakeFiles/*        (CMake build artifacts)
  # ----------------------------------------------------------------------------
  EXCLUDE_PATTERNS=(
    '/usr/*'              # System headers
    '*/gtest/*'           # Google Test framework
    '*/gtests/*'          # Test files (zendnnl/gtests and ai_gtests)
    '*/build/*'           # Build artifacts
    '*/reorder*'
    '*/matmul_native*'
  )
  
  # ============================================================================
  # FILTERING EXECUTION
  # ============================================================================
  # Three modes available:
  # 1. NEGATIVE only: Remove unwanted files (current default)
  # 2. POSITIVE only: Extract only wanted files
  # 3. BOTH: Extract wanted files, then remove unwanted patterns from that set
  # ============================================================================
  
  # Check if INCLUDE_PATHS is defined and not empty
  if [ -n "${INCLUDE_PATHS+x}" ] && [ ${#INCLUDE_PATHS[@]} -gt 0 ]; then
    # POSITIVE filtering is enabled
    echo "Applying POSITIVE filter (extracting specified paths)..."
    lcov --extract "$OUTPUT_INFO" "${INCLUDE_PATHS[@]}" --output-file "temp_extracted.info"
    
    # Check if we should also apply NEGATIVE filtering
    if [ ${#EXCLUDE_PATTERNS[@]} -gt 0 ]; then
      echo "Applying NEGATIVE filter on extracted data..."
      lcov --remove "temp_extracted.info" "${EXCLUDE_PATTERNS[@]}" --output-file "$FILTERED_INFO"
      rm -f "temp_extracted.info"
    else
      mv "temp_extracted.info" "$FILTERED_INFO"
    fi
  else
    # Only NEGATIVE filtering (default mode)
    echo "Applying NEGATIVE filter (removing unwanted files)..."
    lcov --remove "$OUTPUT_INFO" "${EXCLUDE_PATTERNS[@]}" --output-file "$FILTERED_INFO"
  fi
  
  echo "Generating HTML report..."
  genhtml "$FILTERED_INFO" --output-directory "$HTML_DIR"

  echo
  echo "=========================================="
  echo "Code Coverage report for ZENDNN(L) is generated."
  echo "Coverage includes data from:"
  echo ""
  echo "  - AI tests with LOWOHA disabled (--lowoha false)"
  echo "    ZENDNNL_MATMUL_ALGO values: ${ALGO_VALUES_LOWOHA_FALSE[*]}"
  echo ""
  echo "  - AI tests with LOWOHA enabled (--lowoha true)"
  echo "    ZENDNNL_MATMUL_ALGO values: ${ALGO_VALUES_LOWOHA_TRUE[*]}"
  echo ""
  echo "  - Additional run with logging DISABLED for else-branch coverage"
  echo ""
  echo "Logger/Profiler coverage enabled:"
  echo "  - ZENDNNL_ENABLE_PROFILER=1"
  echo "  - All log levels set to 4 (verbose)"
  echo "  - Covers: apilog_info(), log_info(), profile timing, config_manager"
  echo ""
  echo "This covers LOWOHA operators:"
  echo "  - lowoha_operators/matmul/*"
  echo "  - lowoha_operators/embedding_bag/*"
  echo ""
  echo "Open $HTML_DIR/index.html manually in your browser."
  echo "=========================================="
}

#Generate Document
function generate_report() {
  # Call the function to run coverage tests and generate report
  run_and_generate_coverage_report
}

#init()
function init() {
  sanity_check
  setup
  code_coverage_build
  generate_report
}

init
