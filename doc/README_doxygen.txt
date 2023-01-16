#*******************************************************************************
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#*******************************************************************************

Install
    doxygen https://www.tutorialspoint.com/how-to-install-doxygen-on-ubuntu
Install graphviz
    sudo apt-get install -y graphviz
Sample doxygen config file
    https://gist.github.com/ugovaretto/261bd1d16d9a0d2e76ee
Create default config file
    doxygen -g
Run doxygen
    doxygen Doxyfile
Tar the file
    tar -czvf doc/doxygen_out.tar.gz doc/doxygen_out
Minimum changes required to make doxygen work
    http://web.evolbio.mpg.de/~boettcher/other/2016/creating_source_graph.html

    PROJECT_NAME           = "ZenDNN_doxygen"
    OUTPUT_DIRECTORY       = doc/doxygen_out
    BUILTIN_STL_SUPPORT    = YES
    EXTRACT_ALL            = YES
    RECURSIVE              = YES
    HIDE_UNDOC_RELATIONS   = NO
    HAVE_DOT               = YES
    UML_LOOK               = YES
    CALL_GRAPH             = YES
    CALLER_GRAPH           = YES
Documenting the code
    https://www.doxygen.nl/manual/docblocks.html
