# BF16 BRGEMM microkernels
set(sources
    bf16_brgemm_ukernel.cpp
    bf16_gemv_bkc.cpp
    bf16_gemv_bkc_wide.cpp
    bf16_gemv_narrow.cpp)

set(headers
    bf16_brgemm_ukernel.hpp
    bf16_gemv_bkc.hpp
    bf16_gemv_narrow.hpp)

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
