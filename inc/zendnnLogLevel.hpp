/*******************************************************************************
* Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
*******************************************************************************/


#ifndef _ZENDNN_LOG_LEVEL_HPP_
#define _ZENDNN_LOG_LEVEL_HPP_

#include <sstream>
#include <string>
#include <cassert>
#include <cstdlib>

#if !defined(LOG_LEVEL_DEFAULT)
    #define LOG_LEVEL_DEFAULT LOG_LEVEL_ERROR
#endif

using ::std::string;
using ::std::stringstream;

enum LogLevel {
    LOG_LEVEL_DISABLED  = -1,
    LOG_LEVEL_ERROR     =  0,
    LOG_LEVEL_WARNING   =  1,
    LOG_LEVEL_INFO      =  2,
    LOG_LEVEL_VERBOSE0  =  3,
    LOG_LEVEL_VERBOSE1  =  4,
    LOG_LEVEL_VERBOSE2  =  5
};

#define LOG_LEVEL_VERBOSE(n) (LOG_LEVEL_VERBOSE0 + n)

static inline const string logLevelToStr(int logLevel) {
    if (logLevel == LOG_LEVEL_ERROR) {
        return "E";
    }
    else if (logLevel == LOG_LEVEL_WARNING) {
        return "W";
    }
    else if (logLevel == LOG_LEVEL_INFO) {
        return "I";
    }
    else if (logLevel >= LOG_LEVEL_VERBOSE0) {
        stringstream ss;
        ss << "V" << logLevel - LOG_LEVEL_VERBOSE0;
        return ss.str();
    }
    else {
        return "?";
    }
}

static inline int zendnnGetLogLevel(const string &name) {
#ifdef _WIN32
    size_t sz = 0;
    static char *logCstr;
    _dupenv_s(&logCstr, &sz, "ZENDNN_LOG_OPTS");
#else
    static char *logCstr = getenv("ZENDNN_LOG_OPTS");
#endif
    if (!logCstr) {
        return LOG_LEVEL_DEFAULT;
    }
    string logStr(logCstr);

    string namePlusColon(name + ":");
    size_t pos = logStr.find(namePlusColon);
    if (pos == string::npos) {
        namePlusColon = "ALL:";
        pos = logStr.find(namePlusColon);
    }

    if (pos == string::npos) {
        return LOG_LEVEL_DEFAULT;
    }

    size_t epos = pos+ namePlusColon.size();
    char *ep;
    if (epos >= logStr.size()) {
        assert(epos == logStr.size());
    }
    else {
        long x = strtol(logStr.c_str() + epos, &ep, 0);
        size_t fpos = ep - logStr.c_str();
        if (fpos - epos > 0) {
            return x;
        }
    }

    return LOG_LEVEL_DEFAULT;
}

#endif // _ZENDNN_LOG_LEVaEL_HPP_
