# GEMM planners: base blocking + dtype-specific overrides
set(sources
  gemm_planner.cpp
  bf16_gemm_plan.cpp
  fp32_gemm_plan.cpp)

set(headers
  gemm_planner.hpp
  bf16_gemm_plan.hpp
  fp32_gemm_plan.hpp)

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
