/*******************************************************************************
* Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _WIN32

#ifndef ZENDNN_PERF_HPP
#define ZENDNN_PERF_HPP

#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cinttypes>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#define  MAX_SINGLE_EVENTS        16

enum class event_type : int32_t {
    HW_CPU_CYCLES = 0,
    HW_CPU_INSTRUCTIONS,
    HW_CACHE_MISSES_LL,
    HW_CACHE_REF_LL,
    HW_CACHE_READ_MISSES_L1D,
    HW_CACHE_READ_REF_L1D,
    HW_CACHE_MISSES_DTLB,
    HW_CACHE_REF_DTLB,
    HW_BRANCH_MISSES,
    HW_BRANCH_INST,
    HW_STALLED_CYCLES_FRONTEND,
    HW_STALLED_CYCLES_BACKEND,
    SW_PAGE_FAULTS,
    SW_CONTEXT_SWITCHES,
    IPC,
    DTLB_MISS_RATE,
    LLC_MISS_RATE,
    L1D_MISS_RATE,
    BR_MISP_RATE,
    EVENT_TYPE_MAX
};
inline  std::string ToString(event_type v) {
    switch (v) {
    case event_type::HW_CPU_CYCLES:
        return "HW_CPU_CYCLES";
    case event_type::HW_CPU_INSTRUCTIONS:
        return "HW_CPU_INSTRUCTIONS";
    case event_type::HW_CACHE_MISSES_LL:
        return "HW_CACHE_MISSES_LL";
    case event_type::HW_CACHE_REF_LL:
        return "HW_CACHE_REF_LL";
    case event_type::HW_CACHE_READ_MISSES_L1D:
        return "HW_CACHE_READ_MISSES_L1D";
    case event_type::HW_CACHE_READ_REF_L1D:
        return "HW_CACHE_READ_REF_L1D";
    case event_type::HW_CACHE_MISSES_DTLB:
        return "HW_CACHE_MISSES_DTLB";
    case event_type::HW_CACHE_REF_DTLB:
        return "HW_CACHE_REF_DTLB";
    case event_type::HW_BRANCH_MISSES:
        return "HW_BRANCH_MISSES";
    case event_type::HW_BRANCH_INST:
        return "HW_BRANCH_INST";
    case event_type::HW_STALLED_CYCLES_FRONTEND:
        return "HW_STALLED_CYCLES_FRONTEND";
    case event_type::HW_STALLED_CYCLES_BACKEND:
        return "HW_STALLED_CYCLES_BACKEND";
    case event_type::SW_PAGE_FAULTS:
        return "SW_PAGE_FAULTS";
    case event_type::SW_CONTEXT_SWITCHES:
        return "SW_CONTEXT_SWITCHES";
    case event_type::IPC:
        return "IPC";
    case event_type::LLC_MISS_RATE:
        return "LLC_MISS_RATE";
    case event_type::L1D_MISS_RATE:
        return "L1D_MISS_RATE";
    case event_type::BR_MISP_RATE:
        return "BR_MISP_RATE";
    default:
        return "[Unknown event_type]";
    }
}
// single event attribute creation
perf_event_attr event_attr(event_type ev) {
    struct perf_event_attr pea;
    std::memset(&pea, 0, sizeof(struct perf_event_attr));

    switch (ev) {
    case event_type::HW_CPU_CYCLES : {
        pea.type    = PERF_TYPE_HARDWARE;
        pea.config  = PERF_COUNT_HW_CPU_CYCLES;
        break;
    }
    case event_type::HW_CPU_INSTRUCTIONS : {
        pea.type    = PERF_TYPE_HARDWARE;
        pea.config  = PERF_COUNT_HW_INSTRUCTIONS;
        break;
    }
    case event_type::HW_CACHE_MISSES_LL : {
        pea.type    = PERF_TYPE_HARDWARE;
        pea.config  = PERF_COUNT_HW_CACHE_MISSES;
        break;
    }
    case event_type::HW_CACHE_REF_LL : {
        pea.type    = PERF_TYPE_HARDWARE;
        pea.config  = PERF_COUNT_HW_CACHE_REFERENCES;
        break;
    }
    case event_type::HW_CACHE_READ_MISSES_L1D : {
        pea.type    = PERF_TYPE_HW_CACHE;
        pea.config  = (PERF_COUNT_HW_CACHE_L1D) |
                      ((PERF_COUNT_HW_CACHE_OP_READ) << 8) |
                      ((PERF_COUNT_HW_CACHE_RESULT_MISS) << 16);
        break;
    }
    case event_type::HW_CACHE_READ_REF_L1D : {
        pea.type    = PERF_TYPE_HW_CACHE;
        pea.config  = (PERF_COUNT_HW_CACHE_L1D) |
                      ((PERF_COUNT_HW_CACHE_OP_READ) << 8) |
                      ((PERF_COUNT_HW_CACHE_RESULT_ACCESS) << 16);
        break;
    }
    case event_type::HW_CACHE_MISSES_DTLB : {
        pea.type    = PERF_TYPE_HW_CACHE;
        pea.config  = (PERF_COUNT_HW_CACHE_DTLB) |
                      ((PERF_COUNT_HW_CACHE_OP_READ) << 8) |
                      ((PERF_COUNT_HW_CACHE_RESULT_MISS) << 16);
        break;
    }
    case event_type:: HW_CACHE_REF_DTLB : {
        pea.type    = PERF_TYPE_HW_CACHE;
        pea.config  = (PERF_COUNT_HW_CACHE_DTLB) |
                      ((PERF_COUNT_HW_CACHE_OP_READ) << 8) |
                      ((PERF_COUNT_HW_CACHE_RESULT_ACCESS) << 16);
        break;
    }
    case event_type::HW_BRANCH_MISSES : {
        pea.type    = PERF_TYPE_HARDWARE;
        pea.config  = PERF_COUNT_HW_BRANCH_MISSES;
        break;
    }
    case event_type::HW_BRANCH_INST : {
        pea.type    = PERF_TYPE_HARDWARE;
        pea.config  = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
        break;
    }
    case event_type::HW_STALLED_CYCLES_FRONTEND : {
        pea.type    = PERF_TYPE_HARDWARE;
        pea.config  = PERF_COUNT_HW_STALLED_CYCLES_FRONTEND;
        break;
    }
    case event_type::HW_STALLED_CYCLES_BACKEND  : {
        pea.type    = PERF_TYPE_HARDWARE;
        pea.config  = PERF_COUNT_HW_STALLED_CYCLES_BACKEND;
        break;
    }
    case event_type::SW_PAGE_FAULTS : {
        pea.type    = PERF_TYPE_SOFTWARE;
        pea.config  = PERF_COUNT_SW_PAGE_FAULTS;
        break;
    }
    case event_type::SW_CONTEXT_SWITCHES : {
        pea.type    = PERF_TYPE_SOFTWARE;
        pea.config  = PERF_COUNT_SW_CONTEXT_SWITCHES;
        break;
    }
    default : {
        std::cout << "unknown event type" << std::endl;
    }
    }

    pea.size           = sizeof(struct perf_event_attr);
    pea.disabled       = 1;
    pea.exclude_kernel = 1;
    pea.exclude_hv     = 1;
    pea.inherit        = 1;
    pea.inherit_stat   = 1;
    pea.read_format    = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
    return pea;
}

struct  single_event {
    // constructors and destructors
    single_event();
    ~single_event();

    // interface
    bool        open_event(event_type ev);
    bool        start_event(std::string block_id);
    bool        stop_event();
    double       read_event();
    bool        close_event();

    std::string get_error_msg();
    std::string code_block;
    std::string event_in;

    // consts
    const float invalid_result = -1.0;

  private:
    // status bits
    struct  status_t {
        uint8_t  open:1;
        uint8_t  start:1;
        uint8_t  stop:1;
        uint8_t  unused:5;
    } status;

    // read format struct
    struct read_format {
        uint64_t nr;
        // uint64_t time_enabled;
        // uint64_t time_running;
        struct {
            uint64_t value;
            uint64_t id;
        } values[];
    };
    // private functions
    // double _read_counter();

    // make default constructor private
    event_type              ev;
    int                     sfd1, sfd2;
    uint64_t                sid1, sid2;
    char buf[4096];
    struct read_format *rf = (struct read_format *) buf;
    // read_format             prev;
    // read_format             curr;
    int32_t                 error_no;
    std::string             error_msg;
    perf_event_attr         pea_t;
};

single_event::single_event() {
    status.open           = 0;
    status.start          = 0;
    status.stop           = 0;
    error_no              = 0;
}

single_event::~single_event() {
// close_event();
}

bool single_event::open_event(event_type ev_in) {

    if (status.open) {
        return false;
    }
    ev                    = ev_in;
    status.open           = 0;
    status.start          = 0;
    status.stop           = 0;

    switch (ev) {
    case event_type::IPC : {
        pea_t = event_attr(event_type::HW_CPU_INSTRUCTIONS);
        sfd1  = syscall(__NR_perf_event_open, &pea_t, 0, -1, -1, 0);

        pea_t = event_attr(event_type::HW_CPU_CYCLES);
        sfd2  = syscall(__NR_perf_event_open, &pea_t, 0, -1, sfd1, 0);
        break;
    }

    case event_type::LLC_MISS_RATE : {
        pea_t = event_attr(event_type::HW_CACHE_MISSES_LL);
        sfd1  = syscall(__NR_perf_event_open, &pea_t, 0, -1, -1, 0);

        pea_t = event_attr(event_type::HW_CACHE_REF_LL);
        sfd2  = syscall(__NR_perf_event_open, &pea_t, 0, -1, sfd1, 0);
        break;
    }

    case event_type::L1D_MISS_RATE : {
        pea_t = event_attr(event_type::HW_CACHE_READ_MISSES_L1D);
        sfd1  = syscall(__NR_perf_event_open, &pea_t, 0, -1, -1, 0);

        pea_t = event_attr(event_type::HW_CACHE_READ_REF_L1D);
        sfd2  = syscall(__NR_perf_event_open, &pea_t, 0, -1, sfd1, 0);
        break;
    }

    case event_type::BR_MISP_RATE : {
        pea_t = event_attr(event_type::HW_BRANCH_MISSES);
        sfd1  = syscall(__NR_perf_event_open, &pea_t, 0, -1, -1, 0);

        pea_t = event_attr(event_type::HW_BRANCH_INST);
        sfd2  = syscall(__NR_perf_event_open, &pea_t, 0, -1, sfd1, 0);
        break;
    }
    case event_type::DTLB_MISS_RATE : {
        pea_t = event_attr(event_type::HW_CACHE_MISSES_DTLB);
        sfd1  = syscall(__NR_perf_event_open, &pea_t, 0, -1, -1, 0);

        pea_t = event_attr(event_type::HW_CACHE_REF_DTLB);
        sfd2  = syscall(__NR_perf_event_open, &pea_t, 0, -1, sfd1, 0);
        break;
    }

    default : {
        pea_t   = event_attr(ev);
        sfd1 = syscall(__NR_perf_event_open, &pea_t, 0, -1, -1, 0);
        if (sfd1>= 0) {
            ioctl(sfd1, PERF_EVENT_IOC_ID, &sid1);
            status.open = 1;
            status.stop = 1;
            event_in=ToString(ev);
        }
        else {
            error_no  = errno;
            error_msg = strerror(errno);
            return false;
        }
        return true;
    }
    }
    if (sfd1>= 0 && sfd2>=0) {
        ioctl(sfd1, PERF_EVENT_IOC_ID, &sid1);
        ioctl(sfd2, PERF_EVENT_IOC_ID, &sid2);
        status.open = 1;
        status.stop = 1;
        event_in=ToString(ev);
    }
    else {
        error_no  = errno;
        error_msg = strerror(errno);
        return false;
    }
    return true;
}

bool single_event::close_event() {
    if (status.open) {
        close(sfd1);
        close(sfd2);
    }

    status.open           = 0;
    status.start          = 0;
    status.stop           = 0;

    return true;
}
bool single_event::start_event(std::string block_id) {
    if (status.open && status.stop) {
        ioctl(sfd1, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(sfd1, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);

        // if(read(sfd1, &prev, sizeof(prev)) != sizeof(prev)) {
        //     error_no  = errno;
        //     error_msg = strerror(errno);
        //     return false;
        //}
        status.start = 1;
        status.stop  = 0;
        code_block= block_id;
        return true;
    }

    return false;
}

bool single_event::stop_event() {
    if (status.start) {
        //     if(read(sfd, &curr, sizeof(curr)) != sizeof(curr)) {
        //        error_no = errno;
        //         error_msg = strerror(errno);
        //       return false;
        //     }

        ioctl(sfd1, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
        status.start = 0;
        status.stop  = 1;
        return true;
    }

    return false;
}
double single_event::read_event() {
    double val1,val2;
    if (status.stop) {
        if (read(sfd1, buf, sizeof(buf))==-1) {
            error_no = errno;
            error_msg = strerror(errno);
            return -1;
        }
        for (int itr = 0; itr < rf->nr; itr++) {
            if (rf->values[itr].id == sid1) {
                val1 = rf->values[itr].value;
            }
            else {
                (rf->values[itr].id == sid2);
                val2 = rf->values[itr].value;
            }
        }

        if (ev==event_type::IPC) {
            return val1/val2;
        }
        else if (ev==event_type::LLC_MISS_RATE) {
            return val1/val2;
        }
        else if (ev==event_type::L1D_MISS_RATE) {
            return val1/val2;
        }
        else if (ev==event_type::BR_MISP_RATE) {
            return val1/val2;
        }
        else if (ev==event_type::DTLB_MISS_RATE) {
            return val1/val2;
        }
        else {
            return val1;
        }
    }
    return -1;

}

std::string single_event::get_error_msg() {
    return error_msg;
}
#endif
#endif
