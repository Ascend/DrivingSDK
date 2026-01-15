cmake_minimum_required(VERSION 3.19.0)
project(mx_driving)

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/intf.cmake)

option(USE_ARCH35 "Use kernels_arch35 instead of kernels" OFF)

if(USE_ARCH35)
    add_subdirectory(${PROJECT_SOURCE_DIR}/kernels_arch35)
else()
    add_subdirectory(${PROJECT_SOURCE_DIR}/kernels)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/onnx_plugin)
add_subdirectory(${PROJECT_SOURCE_DIR}/mx_driving/csrc)
