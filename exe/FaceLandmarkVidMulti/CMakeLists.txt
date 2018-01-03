#TBB library
include_directories(${TBB_ROOT_DIR}/include)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/Utilities/include)
			
add_executable(FaceLandmarkVidMulti FaceLandmarkVidMulti.cpp)
target_link_libraries(FaceLandmarkVidMulti LandmarkDetector)
target_link_libraries(FaceLandmarkVidMulti Utilities)
target_link_libraries(FaceLandmarkVidMulti dlib)

target_link_libraries(FaceLandmarkVidMulti ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${BLAS_LIBRARIES})

install (TARGETS FaceLandmarkVidMulti DESTINATION bin)
