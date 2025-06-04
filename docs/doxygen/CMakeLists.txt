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

if(ZENDNNL_BUILD_DOXYGEN)
  message(DEBUG "adding doxygen-awesome-css ")
  include(FetchContent)

  FetchContent_Declare(
  doxygen-awesome-css
  URL https://github.com/jothepro/doxygen-awesome-css/archive/refs/heads/main.zip)

  FetchContent_MakeAvailable(doxygen-awesome-css)

  # Save the location the files were cloned into
  # This allows us to get the path to doxygen-awesome.css
  FetchContent_GetProperties(doxygen-awesome-css SOURCE_DIR DOXYGEN_AWESOME_CSS_DIR)

  #set(DOXYGEN_AWESOME_CSS_DIR ${doxygen-awesome-css_SOURCE_DIR})

  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    message(DEBUG "building doxygen documentation...")
    set(DOXYFILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/doxyfile.in")
    set(DOXYFILE_OUT "${CMAKE_CURRENT_BINARY_DIR}/doxyfile.out")
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_INSTALL_PREFIX}/doxygen")

    configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)
    doxygen_add_docs(zendnnl-doxygen ALL
      CONFIG_FILE ${DOXYFILE_OUT})

  list(APPEND DOXYGEN_CLEAN_FILES "${DOXYGEN_OUTPUT_DIRECTORY}")

  set_target_properties(zendnnl-doxygen
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${DOXYGEN_CLEAN_FILES}")

  else()
    message(WARNING "Doxygen not found, unable to build documentation.")
  endif()
else()
  message(STATUS "doxygen documentation build is disabled...")
endif()
