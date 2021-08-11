/*******************************************************************************
* Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include "zendnn_logging.hpp"

namespace zendnn {

namespace cn = std::chrono;

ZendnnLogState::ZendnnLogState(cn::steady_clock::time_point startTime) :
    startTime_(startTime) {
    moduleNames_[ZENDNN_ALGOLOG]  = "ALGO";
    moduleNames_[ZENDNN_CORELOG]  = "CORE";
    moduleNames_[ZENDNN_APILOG]   = "API";
    moduleNames_[ZENDNN_TESTLOG]  = "TEST";
    moduleNames_[ZENDNN_PROFLOG]  = "PROF";
    moduleNames_[ZENDNN_FWKLOG]   = "FWK";


    static_assert(ZENDNN_NUM_LOG_MODULES == 6,
                  "Need to update moduleNames_ initialization");

    for (int mod = 0; mod < ZENDNN_NUM_LOG_MODULES; mod++) {
        auto name = moduleNames_.at(mod);
        int lvl = zendnnGetLogLevel(name);
        //std::cout << "mod: " << mod << "\n";
        //std::cout << "name: " << name << "\n";
        //std::cout << "lvl: " << lvl << "\n";
        moduleLevels_.at(mod) = lvl;
    }

    log = &std::cout;
}

ZendnnLogState *
_zendnnGetLogState(void) {
    static ZendnnLogState logState(cn::steady_clock::now());
    return &logState;
}

} //namespace zendnn
