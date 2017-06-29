find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(VTK REQUIRED)
find_package(Eigen REQUIRED)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED )
#setting caffe distribute dir
# TODO: add caffe requirement in the documentation
set(Caffe_DISTRIBUTE_DIR "/home/gautam/gsoc/caffe/distribute")
find_package(Caffe REQUIRED)
# Glog is required for caffe_xor, and mnist_train examples
find_package(Glog REQUIRED)
ADD_DEFINITIONS(
    -std=c++11 
	${Caffe_DEFINITIONS}
)
#add caffe include dirs
include_directories("${OD_SOURCE_DIR}" ${EIGEN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OD_SOURCE_DIR}/3rdparty/SiftGPU/src/SiftGPU ${Caffe_INCLUDE_DIRS} )


ADD_DEFINITIONS(${Caffe_DEFINITIONS})
