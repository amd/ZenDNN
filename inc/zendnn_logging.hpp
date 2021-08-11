/*******************************************************************************
* Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <array>
#include <chrono>
#include "zendnnLogLevel.hpp"

//#define LOG_LEVEL_DEFAULT LOG_LEVEL_WARNING

namespace zendnn {

namespace cn = std::chrono;

enum ZendnnLogModule {
    ZENDNN_ALGOLOG,
    ZENDNN_CORELOG,
    ZENDNN_APILOG,
    ZENDNN_TESTLOG,
    ZENDNN_PROFLOG,
    ZENDNN_FWKLOG,
    ZENDNN_NUM_LOG_MODULES
};

struct ZendnnLogState {
    ZendnnLogState(cn::steady_clock::time_point startTime);
    cn::steady_clock::time_point startTime_;
    std::array<int, ZENDNN_NUM_LOG_MODULES> moduleLevels_;
    std::array<const char *, ZENDNN_NUM_LOG_MODULES> moduleNames_;
    std::ofstream logFIle;
    std::ostream *log;
    //std::ios iosDefaultState;
};

ZendnnLogState *_zendnnGetLogState(void);

static inline void
_zendnnLogMessageR(ZendnnLogState *logState) {
    *logState->log << "\n";
}

template <typename T, typename... Ts>
static inline void
_zendnnLogMessageR(ZendnnLogState *logState, T arg0, Ts... arg1Misc) {
    *logState->log << arg0;
    _zendnnLogMessageR(logState, arg1Misc...);
}

template <typename... Ts>
static inline void
_zendnnLogMessage(LogLevel level, ZendnnLogModule mod, Ts... vs) {
    auto logState = _zendnnGetLogState();
    auto now_t = cn::steady_clock::now();
    auto us = cn::duration_cast<cn::microseconds>
              (now_t - logState->startTime_).count();
    auto moduleName = logState->moduleNames_.at(mod);
    auto logLevelStr = logLevelToStr(level);
    float secs = (float)us/ 1000000.0f;

    char logHdr[32];
    std::snprintf(logHdr, sizeof(logHdr),
                  "[%s:%s][%.6f] ",
                  moduleName,
                  logLevelStr.c_str(),
                  secs);

    _zendnnLogMessageR(logState, logHdr, vs...);
}

#define zendnnLogAtLevel(mod, level, ...) do {                  \
    if (level <= _zendnnGetLogState()->moduleLevels_.at(mod)) { \
        _zendnnLogMessage(level, mod, ##__VA_ARGS__);            \
    }                                                           \
} while (0)

#define zendnnInfo(mod, ...)    zendnnLogAtLevel(mod, LOG_LEVEL_INFO, ##__VA_ARGS__)
#define zendnnWarn(mod, ...)    zeddnnLogAtLevel(mod, LOG_LEVEL_WARNING, ##__VA_ARGS__)
#define zendnnError(mod, ...)   zendnnLogAtLevel(mod, LOG_LEVEL_ERROR, ##__VA_ARGS__)
#define zendnnVerbose(mod, ...) zendnnLogAtLevel(mod, LOG_LEVEL_VERBOSE0, ##__VA_ARGS__)


template <typename... Ts>
static void
conditionFailed(const char *cond, const char *file, int line,
                ZendnnLogModule mod, Ts... vs) {
    zendnnError(mod, "FAILED: ", cond,  "\n",
                "FILE: ", file, "\n",
                "LINE: ", line, "\n",
                "MESSAGE: ", vs...);
}


/* Checks weather he given condition is satisfied, else print error message and
 * return error code in debug build and do nothing in release build
 */
#define ZENDNN_CHECK(CONDITION, MOD, ...) do {                              \
    if (!(CONDITION)) {                                                      \
        conditionFailed(#CONDITION, __FILE__, __LINE__, MOD, ##__VA_ARGS__); \
    }                                                                        \
} while (0)


}
