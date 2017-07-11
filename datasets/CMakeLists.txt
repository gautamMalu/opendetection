set(SUBSYS_NAME datasets)
set(LIB_NAME od_${SUBSYS_NAME})
set(SUBSYS_DESC "Different Datasets Classes")

set(SUBSYS_DEPS  od_common ${OpenCV_LIBS} ${GLOG_LIBRARIES} ${Caffe_LIBRARIES} ${CUDA_cublas_LIBRARY} 
${Boost_SYSTEM_LIBRARY_RELEASE} ${LMDB_LIBRARIES} ${GFLAGS_LIBRARIES} ${PROTOBUF_LIBRARIES})

include_directories(${OpenCV_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS} ${CUDA_TOOLKIT_INCLUDE} ${LMDB_INCLUDE_DIR} ${GFLAGS_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIR})


set(build TRUE)
ADD_DEFINITIONS(
    -std=c++11 
	${Caffe_DEFINITIONS}
)


if(build)

    set(incs
        "common.h"
	"ODDatasetMNIST.h"
	"ODDatasetCIFAR.h"
	"ODDatasetList.h"
	)

    set(srcs
        "ODDatasetMNIST.cpp"
        "ODDatasetCIFAR.cpp"
        "ODDatasetList.cpp"
	"ODDatasetFolder.cpp"
        )


    OD_ADD_LIBRARY_ALL("${SUBSYS_NAME}" SRCS ${srcs} INCS ${incs})
    install(FILES ${incs} DESTINATION ${OD_INSTALL_INCLUDE_DIR}/${SUBSYS_NAME} COMPONENT ${LIB_NAME})

    if(SUBSYS_DEPS)
      target_link_libraries("${LIB_NAME}" ${SUBSYS_DEPS} )
    endif(SUBSYS_DEPS)

endif(build)
