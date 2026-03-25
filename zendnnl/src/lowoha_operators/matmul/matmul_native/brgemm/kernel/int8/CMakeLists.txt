# INT8 VNNI microkernels
set(sources
    int8_gemv_bkc.cpp
    int8_gemv_bkc_wide.cpp
    int8_brgemm_ukernel.cpp)

set(headers
    int8_gemv_bkc.hpp
    int8_brgemm_ukernel.hpp)

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
