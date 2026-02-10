find_package(CUDA REQUIRED)

add_executable(buildindex buildindex.cpp)
target_link_libraries(buildindex ${CUDA_LIBRARIES})

add_executable(search search.cpp)
target_link_libraries(search ${CUDA_LIBRARIES})

add_executable(search_multi search_multi.cpp)
target_link_libraries(search_multi ${CUDA_LIBRARIES})