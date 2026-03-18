find_package(CUDA REQUIRED)

add_executable(buildindex buildindex.cpp)
target_link_libraries(buildindex ${CUDA_LIBRARIES})

add_executable(search search.cpp)
target_link_libraries(search ${CUDA_LIBRARIES})

add_executable(search_multi search_multi.cpp)
target_link_libraries(search_multi ${CUDA_LIBRARIES})

if(ENABLE_FAISS)
	find_path(FAISS_INCLUDE_DIR faiss/IndexFlat.h)
	find_library(FAISS_LIBRARY faiss)
	find_package(BLAS REQUIRED)
	find_package(LAPACK REQUIRED)
	find_package(OpenMP REQUIRED)

	if(FAISS_INCLUDE_DIR AND FAISS_LIBRARY)
		add_executable(pq_encode pq_encode.cpp)
		target_include_directories(pq_encode PRIVATE ${FAISS_INCLUDE_DIR})
		target_link_libraries(pq_encode PRIVATE ${FAISS_LIBRARY} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})

		add_executable(pq_manual_distance pq_manual_distance.cpp)
		target_include_directories(pq_manual_distance PRIVATE ${FAISS_INCLUDE_DIR})
		target_link_libraries(pq_manual_distance PRIVATE ${FAISS_LIBRARY} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)

		message(STATUS "FAISS support enabled: pq_encode and pq_manual_distance targets available")
	else()
		message(WARNING "ENABLE_FAISS=ON but FAISS headers/library were not found; pq targets will not be built")
	endif()
endif()