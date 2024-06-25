/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/
#include "zentpp_memory.hpp"

namespace zendnn {
namespace tpp {

ZenTensor empty_like(const ZenTensor& atensor) {
    auto md      = atensor.get_desc();
    auto aengine = atensor.get_engine();

    return ZenTensor(md, aengine);
}

ZenTensor empty_tensor(std::vector<int64_t> asize, ZenDType adt, engine aengine) {
    memory::format_tag tag = memory::format_tag::any;

    switch(asize.size()) {
    case 1: tag = memory::format_tag::a; break;
    case 2: tag = memory::format_tag::ab; break;
    case 3: tag = memory::format_tag::abc; break;
    case 4: tag = memory::format_tag::abcd; break;
    case 5: tag = memory::format_tag::abcde; break;
    case 6: tag = memory::format_tag::abcdef; break;
    case 7: tag = memory::format_tag::abcdefg; break;
    case 8: tag = memory::format_tag::abcdefgh; break;
    case 9: tag = memory::format_tag::abcdefghi; break;
    }

    memory::desc md(asize, adt, tag);

    return ZenTensor(md, aengine);
}

ZenTensor zero_tensor(std::vector<int64_t> asize, ZenDType adt, engine aengine) {
    memory::format_tag tag = memory::format_tag::any;

    switch(asize.size()) {
    case 1: tag = memory::format_tag::a; break;
    case 2: tag = memory::format_tag::ab; break;
    case 3: tag = memory::format_tag::abc; break;
    case 4: tag = memory::format_tag::abcd; break;
    case 5: tag = memory::format_tag::abcde; break;
    case 6: tag = memory::format_tag::abcdef; break;
    case 7: tag = memory::format_tag::abcdefg; break;
    case 8: tag = memory::format_tag::abcdefgh; break;
    case 9: tag = memory::format_tag::abcdefghi; break;
    }

    memory::desc md(asize, adt, tag);

    ZenTensor rtensor(md, aengine);

    return rtensor;
}

}
}
