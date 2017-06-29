project (mnist_train)

include_directories(${OpenCV_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS} ${CUDA_TOOLKIT_INCLUDE})
#ADD_DEFINITIONS(${Caffe_DEFINITIONS})
set(SOURCES od_cnn_mnist_train_simple.cpp)
add_executable(mnist_train ${SOURCES})

target_link_libraries(mnist_train ${OpenCV_LIBRARIES} ${GLOG_LIBRARIES} ${Caffe_LIBRARIES} ${CUDA_cublas_LIBRARY} ${Boost_SYSTEM_LIBRARY_RELEASE} od_common od_global_image_detector)


