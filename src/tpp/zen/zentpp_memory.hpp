/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <vector>
#include "zendnn.h"
#include "zendnn.hpp"

namespace zendnn {
namespace tpp {

using ZenDType = typename memory::data_type;

/*
 * tpp specific tensor. this is wrapper around zendnn memory class.
 */

class ZenTensor : public memory{
public:
    // constructors
    ZenTensor() : memory(), defined{true}, contiguous{true} {
        // create an empty tensor
    }

    ZenTensor(const memory::desc& md, const engine& aengine) :
        memory(md, aengine), defined{true}, contiguous{true} {
    }

    ZenTensor(const memory::desc& md, const engine& aengine, void* handle) :
        memory(md, aengine, handle), defined{true}, contiguous{true} {
    }

    // tensor dim
    int64_t dimension() const {
        return get_desc().dims().size();
    }

    // number of elements
    int64_t numel() const {
        dims size_vector = get_desc().dims();

        if (size_vector.size()) {
            int64_t size = 1;
            for (auto sz: size_vector)
                size *= sz;

            return size;
        }

        return 0;
    }

    // sizes of the tensor
    std::vector<int64_t> sizes() const {
        return get_desc().dims();
    }

    // is defined
    bool is_defined() const {
        return true;
    }

    // is contiguous
    bool is_contiguous() const {
        return true;
    }

    template<typename T>
    T* data_ptr() {
        return static_cast<T*>(memory::get_data_handle());
    }

    ZenDType dtype() const {
        return get_desc().data_type();
    }

    ZenTensor new_empty(std::vector<int64_t> asize) const {
        ZenDType dt      = get_desc().data_type();
        engine   aengine = get_engine();

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

        memory::desc md(asize, dt, tag);

        return ZenTensor(md, aengine);
    }

private:
    bool defined;
    bool contiguous;
};

//supported functions
ZenTensor empty_like(const ZenTensor& atensor);
ZenTensor empty_tensor(std::vector<int64_t> asize, ZenDType adt, engine aengine);
ZenTensor zero_tensor(std::vector<int64_t> asize, ZenDType adt, engine aengine);

} //tpp
} //zendnn
