file(GLOB ONNX_PLUGIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
if(BUILD_STAGE EQUAL 1)
  if(${ENABLE_ONNX})
    # check if protoc exists
    find_program(PROTOC_EXECUTABLE protoc)
    if(NOT PROTOC_EXECUTABLE)
      message(WARNING "protoc not found")
    else()
      if(CANN_PATHS)
        if(${ARCH} STREQUAL "aarch64")
          protobuf_generate(
            PROTO_FILE ${CANN_PATHS}/aarch64-linux/include/proto/ge_onnx.proto
            OUT_DIR ${ASCEND_AUTOGEN_PATH})
        else()
          protobuf_generate(
            PROTO_FILE ${CANN_PATHS}/x86_64-linux/include/proto/ge_onnx.proto
            OUT_DIR ${ASCEND_AUTOGEN_PATH})
        endif()
      else()
        protobuf_generate(
          PROTO_FILE ${ASCEND_CANN_PACKAGE_PATH}/include/proto/ge_onnx.proto
          OUT_DIR ${ASCEND_AUTOGEN_PATH})
      endif()

      add_library(cust_onnx_parsers SHARED ${ONNX_PLUGIN_SRC})
      target_compile_options(
        cust_onnx_parsers
        PRIVATE -O2 -Werror -Wno-deprecated-declarations
                -Dgoogle=ascend_private "-fno-common" "-fno-strict-aliasing")
      target_link_libraries(cust_onnx_parsers PRIVATE intf_pub)
      target_include_directories(
        cust_onnx_parsers PRIVATE ${PROJECT_SOURCE_DIR}/include
                                  ${ASCEND_AUTOGEN_PATH})

      install_target(TRG cust_onnx_parsers DST
                     packages/vendors/${vendor_name}/framework/onnx/)
    endif()
  endif()
endif()
