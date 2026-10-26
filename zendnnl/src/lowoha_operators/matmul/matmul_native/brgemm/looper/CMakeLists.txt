# BRGEMM loopers (tile loops + OMP) and GEMV fast path
set(sources
    bf16_brgemm_looper.cpp
  bf16_gemv_direct.cpp
  int8_gemv_direct.cpp
  int8_brgemm_looper.cpp
  fp32_brgemm_looper.cpp
  native_brgemm_looper.cpp
  native_int8_call.cpp
  native_int8_channel_tile.cpp
  native_int8_group_tile.cpp
  native_int8_tile.cpp
  native_partition.cpp)

set(headers
    bf16_brgemm_looper.hpp
  bf16_gemv_direct.hpp
  int8_gemv_direct.hpp
  int8_brgemm_looper.hpp
  fp32_brgemm_looper.hpp
  native_brgemm_looper.hpp
  native_int8_call.hpp
  native_int8_tile.hpp
  native_int8_tile_internal.hpp
  native_partition.hpp)

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
