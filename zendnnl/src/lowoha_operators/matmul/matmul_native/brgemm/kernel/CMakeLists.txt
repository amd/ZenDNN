add_subdirectory(bf16)
add_subdirectory(fp32)
add_subdirectory(int8)

# Native BRGEMM kernel capabilities
set(sources
    native_kernel_registry.cpp)

set(headers
    native_kernel_registry.hpp)

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
