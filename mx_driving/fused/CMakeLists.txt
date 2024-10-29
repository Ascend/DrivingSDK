if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ops/kernels)
  add_subdirectory(ops/kernels)
endif()

if (${ENABLE_ONNX} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ops/onnx)
  add_subdirectory(ops/onnx/plugin)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ops/csrc)
  add_subdirectory(ops/csrc)
endif()
