# GEMM loopers (tile loops + OMP)
set(sources
    bf16_gemm_looper.cpp
  fp32_gemm_looper.cpp)

set(headers
    bf16_gemm_looper.hpp
  fp32_gemm_looper.hpp)

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
