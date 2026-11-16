# GEMV kernels of M = 1 calls over the native layout
set(sources
    native_gemv_bf16.cpp
    native_gemv_f16_ph.cpp
    native_gemv_float.cpp
    native_gemv_int8.cpp)

set(headers
    native_gemv_ukernel.hpp)

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
