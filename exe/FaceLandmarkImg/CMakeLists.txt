#TBB library
include_directories(${TBB_ROOT_DIR}/include)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/FaceAnalyser/include)	
include_directories(../../lib/local/GazeAnalyser/include)	
include_directories(../../lib/local/Utilities/include)	
			
add_executable(FaceLandmarkImg FaceLandmarkImg.cpp)
target_link_libraries(FaceLandmarkImg LandmarkDetector)
target_link_libraries(FaceLandmarkImg FaceAnalyser)
target_link_libraries(FaceLandmarkImg GazeAnalyser)
target_link_libraries(FaceLandmarkImg Utilities)
target_link_libraries(FaceLandmarkImg dlib)

target_link_libraries(FaceLandmarkImg ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES}  ${BLAS_LIBRARIES})

install (TARGETS FaceLandmarkImg DESTINATION bin)
