# oneDNN backend kernel
set(sources
    onednn_kernel.cpp
    onednn_utils.cpp
    onednn_execute.cpp)

set(headers
    onednn_kernel.hpp
    onednn_utils.hpp
    onednn_execute.hpp)

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
