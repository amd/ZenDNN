# BRGEMM planner: the native BRGEMM plan
set(sources
  native_config.cpp
  native_planner.cpp
  native_tuning_file.cpp)

set(headers
  native_config.hpp
  native_planner.hpp
  native_tuning_file.hpp
  native_tuning_keys.hpp)

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
