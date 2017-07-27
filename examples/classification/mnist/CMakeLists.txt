OD_ADD_EXAMPLE(convert_mnist FILES convert_mnist_data.cpp
                LINK_WITH od_datasets INCLUDES ${CUDA_TOOLKIT_INCLUDE} )

OD_ADD_EXAMPLE(train_mnist FILES train_mnist.cpp
                LINK_WITH od_global_image_detector INCLUDES ${CUDA_TOOLKIT_INCLUDE})

OD_ADD_EXAMPLE(classify_mnist FILES classify_mnist.cpp
                LINK_WITH od_global_image_detector od_common INCLUDES ${CUDA_TOOLKIT_INCLUDE})
