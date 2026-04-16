# Cache infrastructure
set(public_headers
    lru_cache.hpp
    zendnnl_key.hpp)

set(private_headers
    lowoha_cache.hpp)

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  target_sources(zendnnl_archive
    PUBLIC  FILE_SET zendnnl_headers
            TYPE HEADERS FILES ${public_headers}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${private_headers})
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  target_sources(zendnnl
    PUBLIC  FILE_SET zendnnl_headers
            TYPE HEADERS FILES ${public_headers}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${private_headers})
endif()
