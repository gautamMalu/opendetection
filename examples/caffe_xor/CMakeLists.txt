project (caffe_xor)

include_directories(${OpenCV_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS} ${CUDA_TOOLKIT_INCLUDE})
ADD_DEFINITIONS(${Caffe_DEFINITIONS})
set(SOURCES cpp_xor.cpp)
add_executable(caffe_xor ${SOURCES})

target_link_libraries(caffe_xor ${OpenCV_LIBRARIES} ${GLOG_LIBRARIES} ${Caffe_LIBRARIES} ${CUDA_cublas_LIBRARY} ${Boost_SYSTEM_LIBRARY_RELEASE})


