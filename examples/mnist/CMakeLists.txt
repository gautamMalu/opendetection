project (mnist)

include_directories(${OpenCV_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS} ${CUDA_TOOLKIT_INCLUDE} 
# adding dependecies for convert mnist data
${LMDB_INCLUDE_DIR} ${GFLAGS_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIR})

add_executable(convert_mnist_data convert_mnist_data.cpp)
add_executable(train_mnist train_mnist.cpp)
add_executable(classify_mnist classify_mnist.cpp)

target_link_libraries(train_mnist od_global_image_detector ${OpenCV_LIBRARIES} ${GLOG_LIBRARIES} ${Caffe_LIBRARIES} 
${CUDA_cublas_LIBRARY} ${Boost_SYSTEM_LIBRARY_RELEASE})

target_link_libraries(convert_mnist_data od_datasets ${OpenCV_LIBRARIES} ${GLOG_LIBRARIES} ${Caffe_LIBRARIES} 
${CUDA_cublas_LIBRARY} ${Boost_SYSTEM_LIBRARY_RELEASE} ${LMDB_LIBRARIES} ${GFLAGS_LIBRARIES}
${PROTOBUF_LIBRARIES})

target_link_libraries(classify_mnist od_generic_classifier od_common ${OpenCV_LIBRARIES} ${GLOG_LIBRARIES} ${Caffe_LIBRARIES} 
${CUDA_cublas_LIBRARY} ${Boost_SYSTEM_LIBRARY_RELEASE})
