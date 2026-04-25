find_package(CUDA REQUIRED)
enable_language(CUDA)

if(ENABLE_FAISS)
	find_path(FAISS_INCLUDE_DIR faiss/IndexFlat.h)
	find_library(FAISS_LIBRARY faiss)

add_executable(buildindex buildindex.cpp)
target_include_directories(buildindex PRIVATE ${FAISS_INCLUDE_DIR})
target_link_libraries(buildindex ${CUDA_LIBRARIES})

add_executable(search search.cpp)
target_include_directories(search PRIVATE ${FAISS_INCLUDE_DIR})
target_link_libraries(search ${CUDA_LIBRARIES})

add_executable(search_multi search_multi.cpp)
target_include_directories(search_multi PRIVATE ${FAISS_INCLUDE_DIR})
target_link_libraries(search_multi ${CUDA_LIBRARIES})

else()

add_executable(buildindex buildindex.cpp)
target_link_libraries(buildindex ${CUDA_LIBRARIES})

add_executable(search search.cpp)
target_link_libraries(search ${CUDA_LIBRARIES})

add_executable(search_multi search_multi.cpp)
target_link_libraries(search_multi ${CUDA_LIBRARIES})

endif()

if(ENABLE_FAISS)
	find_package(BLAS REQUIRED)
	find_package(LAPACK REQUIRED)
	find_package(OpenMP REQUIRED)

		add_executable(pq_encode pq_encode.cpp)
		target_include_directories(pq_encode PRIVATE ${FAISS_INCLUDE_DIR})
		target_link_libraries(pq_encode PRIVATE ${FAISS_LIBRARY} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})

		add_executable(pq_compress_only pq_compress_only.cpp)
		target_include_directories(pq_compress_only PRIVATE ${FAISS_INCLUDE_DIR})
		target_link_libraries(pq_compress_only PRIVATE ${FAISS_LIBRARY} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)

		add_executable(pq_manual_distance pq_manual_distance.cpp)
		target_include_directories(pq_manual_distance PRIVATE ${FAISS_INCLUDE_DIR})
		target_link_libraries(pq_manual_distance PRIVATE ${FAISS_LIBRARY} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)

		# TODO: gpu_pq_encode and gpu_irangegraph_search to be added later
		# add_executable(gpu_pq_encode gpu_pq_encode.cu)
		# target_include_directories(gpu_pq_encode PRIVATE ${FAISS_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/cude_version)
		# target_link_libraries(gpu_pq_encode PRIVATE ${FAISS_LIBRARY} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})
		# set_target_properties(gpu_pq_encode PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
		
		# GPU iRangeGraph Greedy Search (based on hello_clean.cu from cude_version)
		# add_executable(gpu_irangegraph_search gpu_irangegraph_search.cu)
		# target_include_directories(gpu_irangegraph_search PRIVATE ${PROJECT_SOURCE_DIR}/cude_version)
		# target_link_libraries(gpu_irangegraph_search PRIVATE ${CUDA_LIBRARIES})
		# set_target_properties(gpu_irangegraph_search PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()