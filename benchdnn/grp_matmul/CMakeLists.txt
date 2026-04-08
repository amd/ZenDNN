set(grp_matmul_sources
grp_matmul_benchdnn.cpp
grp_matmul_utils.cpp)

add_library(grp_matmul_benchdnn STATIC ${grp_matmul_sources})

if(TARGET zendnnl::zendnnl_archive)
  target_link_libraries(grp_matmul_benchdnn zendnnl::zendnnl_archive)
elseif(TARGET zendnnl::zendnnl)
  target_link_libraries(grp_matmul_benchdnn zendnnl::zendnnl)
else()
  message(FATAL_ERROR "No zendnnl library target found for grp_matmul_benchdnn!")
endif()
