OD_ADD_EXAMPLE(convert_cub FILES convert_cub_data.cpp
                LINK_WITH od_datasets INCLUDES ${CUDA_TOOLKIT_INCLUDE} )

OD_ADD_EXAMPLE(train_cub FILES train_cub.cpp
                LINK_WITH od_global_image_detector INCLUDES ${CUDA_TOOLKIT_INCLUDE})

OD_ADD_EXAMPLE(classify_cub FILES classify_cub.cpp
                LINK_WITH od_global_image_detector od_common INCLUDES ${CUDA_TOOLKIT_INCLUDE})
