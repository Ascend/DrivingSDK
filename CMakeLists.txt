cmake_minimum_required(VERSION 3.19.0)
project(mx_driving)

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/intf.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/kernels)
add_subdirectory(${PROJECT_SOURCE_DIR}/onnx_plugin)
add_subdirectory(${PROJECT_SOURCE_DIR}/mx_driving/csrc)
