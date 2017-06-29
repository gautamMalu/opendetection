###############################################################################
# Find Caffe
#
# This sets the following variables:
# Caffe_FOUND - True if Eigen was found.
# Caffe_INCLUDE_DIRS - Directories containing the Eigen include files.
# Caffe_LIBRARIES - Caffe library location.

unset(Caffe_FOUND)

find_path(Caffe_INCLUDE_DIRS NAMES caffe/caffe.hpp caffe/common.hpp caffe/net.hpp caffe/proto/caffe.pb.h caffe/util/io.hpp caffe/vision_layers.hpp
  HINTS
  ${Caffe_DISTRIBUTE_DIR}/include)


find_library(Caffe_LIBRARIES NAMES caffe
  HINTS
  ${Caffe_DISTRIBUTE_DIR}/lib)

message("lib_dirs:${Caffe_LIBRARIES}")

if(Caffe_LIBRARIES AND Caffe_INCLUDE_DIRS)
    set(Caffe_FOUND 1)
endif()
