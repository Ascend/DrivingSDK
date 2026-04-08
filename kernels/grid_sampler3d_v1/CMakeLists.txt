# 算子 grid_sampler3d_v1 的CMake配置

if(USE_ARCH35)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_host/arch35)
        file(GLOB OP_HOST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_host/arch35/*.cpp
             ${CMAKE_CURRENT_SOURCE_DIR}/op_host/arch35/*.h)
    else()
        file(GLOB OP_HOST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.cpp
             ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.h)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/arch35)
        file(GLOB OP_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/arch35/*.cpp
             ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/arch35/*.h)
    else()
        file(GLOB OP_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/*.cpp
             ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/*.h)
    endif()
else()
    file(GLOB OP_HOST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.h)
    file(GLOB OP_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/*.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/*.h)
endif()

set(ASCEND_HOST_SRC ${ASCEND_HOST_SRC} ${OP_HOST_SRC} PARENT_SCOPE)
set(ASCEND_KERNEL_SRC ${ASCEND_KERNEL_SRC} ${OP_KERNEL_SRC} PARENT_SCOPE)
