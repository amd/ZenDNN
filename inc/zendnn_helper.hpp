/*******************************************************************************
* Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#pragma once

#include <iostream>


namespace zendnn {

/// Read an integer from the environment variable
/// Return default_value if the environment variable is not defined, otherwise
/// return actual value.
inline int zendnn_getenv_int(const char *name, int default_value = 0) {
    char* val = std::getenv(name);
    return val == NULL ? default_value : atoi(val);
}

/// Read an float from the environment variable
/// Return default_value if the environment variable is not defined, otherwise
/// return actual value.
inline float zendnn_getenv_float(const char *name, float default_value = 0.0f) {
    char* val = std::getenv(name);
    return val == NULL ? default_value : atof(val);
}

/// Read an string from the environment variable
/// Return empty string "" if the environment variable is not defined, otherwise
/// return actual value.
inline std::string zendnn_getenv_string(const char *name,
        std::string default_value = "") {
    char* val = std::getenv(name);
    return val == NULL ? default_value : std::string(val);
}

}
