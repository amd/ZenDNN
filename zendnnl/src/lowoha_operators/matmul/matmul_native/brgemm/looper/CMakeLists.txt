# BRGEMM loopers (tile loops + OMP) and the loopers of the GEMV kernels
set(sources
  native_brgemm_looper.cpp
  native_int8_call.cpp
  native_int8_channel_tile.cpp
  native_int8_group_tile.cpp
  native_int8_tile.cpp
  native_partition.cpp)

set(headers
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
