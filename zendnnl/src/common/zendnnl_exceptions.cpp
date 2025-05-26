/********************************************************************************
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

#include "zendnnl_exceptions.hpp"

namespace zendnnl {
namespace error_handling {

exception_t::exception_t(std::string err_msg) {
    what_msg = "\nexception: " + err_msg;
}

exception_t::exception_t(const char* err_msg):
    exception_t{std::string(err_msg)}{
}

exception_t::exception_t(const char* file, int line, std::string err_msg) {
    what_msg = "\nexception at: ";
    what_msg += "[";
    what_msg += std::string(file);
    what_msg += "] [";
    what_msg += std::to_string(line);
    what_msg += "] ";
    what_msg = what_msg + err_msg;
}

exception_t::exception_t(const char* file, int line, const char* err_msg):
    exception_t{file, line, std::string(err_msg)} {
}

const char* exception_t::what() const noexcept {
    return what_msg.c_str();
}

} //error_handling
} //zendnnl
