# INT8 VNNI microkernels, s8 and s4 weights
set(sources
    int8_gemv_bkc.cpp
    int8_gemv_bkc_wide.cpp
    native_int4_ukernel.cpp
    native_int8_ukernel.cpp)

set(headers
    int8_gemv_bkc.hpp
    native_int4_ukernel.hpp
    native_int8_ukernel.hpp)

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  target_sources(zendnnl_archive
    PRIVATE ${sources}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${headers})
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  target_sources(zendnnl
    PRIVATE ${sources}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${headers})
endif()
