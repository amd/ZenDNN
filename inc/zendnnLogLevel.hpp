/*******************************************************************************
* Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/


#ifndef _ZENDNN_LOG_LEVEL_HPP_
#define _ZENDNN_LOG_LEVEL_HPP_

#include <sstream>
#include <string>
#include <cassert>

#if !defined(LOG_LEVEL_DEFAULT)
    #define LOG_LEVEL_DEFAULT LOG_LEVEL_ERROR
#endif

enum LogLevel {
    LOG_LEVEL_DISABLED  = -1,
    LOG_LEVEL_ERROR     =  0,
    LOG_LEVEL_WARNING   =  1,
    LOG_LEVEL_INFO      =  2,
    LOG_LEVEL_VERBOSE0  =  3
};

#define LOG_LEVEL_VERBOSE(n) (LOG_LEVEL_VERBOSE0 + n)

static inline const std::string logLevelToStr(int logLevel) {
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
        std::stringstream ss;
        ss << "V" << logLevel - LOG_LEVEL_VERBOSE0;
        return ss.str();
    }
    else {
        return "?";
    }
}

static inline int zendnnGetLogLevel(const std::string &name) {
    static char *logCstr = getenv("ZENDNN_LOG_OPTS");
    if (!logCstr) {
        return LOG_LEVEL_DEFAULT;
    }
    std::string logStr(logCstr);

    size_t pos, epos;

    std::string namePlusColon(name + ":");
    pos = logStr.find(namePlusColon);
    if (pos == std::string::npos) {
        namePlusColon = "ALL:";
        pos = logStr.find(namePlusColon);
    }

    if (pos == std::string::npos) {
        return LOG_LEVEL_DEFAULT;
    }

    epos = pos+ namePlusColon.size();
    long x;
    char *ep;
    if (epos >= logStr.size()) {
        assert(epos == logStr.size());
    }
    else {
        x = strtol(logStr.c_str() + epos, &ep, 0);
        size_t fpos = ep - logStr.c_str();
        if (fpos - epos > 0) {
            return x;
        }
    }

    return LOG_LEVEL_DEFAULT;
}

#endif // _ZENDNN_LOG_LEVaEL_HPP_
