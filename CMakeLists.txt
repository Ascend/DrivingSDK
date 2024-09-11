cmake_minimum_required(VERSION 3.16.0)
project(opp)
set(CMAKE_COMPILE ${CMAKE_CXX_COMPILER})

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/intf.cmake)

set(MX_DRIVING_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mx_driving)
add_subdirectory(${MX_DRIVING_DIR}/common)
add_subdirectory(${MX_DRIVING_DIR}/preprocess)
add_subdirectory(${MX_DRIVING_DIR}/fused)
add_subdirectory(${MX_DRIVING_DIR}/point)
add_subdirectory(${MX_DRIVING_DIR}/detection)
add_subdirectory(${MX_DRIVING_DIR}/spconv)

opbuild(OPS_SRC ${ASCEND_HOST_SRC} OUT_DIR ${ASCEND_AUTOGEN_PATH})

add_library(cust_op_proto SHARED ${ASCEND_HOST_SRC}
                                 ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(cust_op_proto PRIVATE -fvisibility=hidden)
target_link_libraries(
  cust_op_proto
  PRIVATE intf_pub
          exe_graph
          register
          tiling_api
          -Wl,--whole-archive
          rt2_registry
          -Wl,--no-whole-archive)
set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME cust_opsproto_rt2.0)
install_target(
  TRG cust_op_proto DST
  packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
install_file(TRG cust_op_proto SRC ${ASCEND_AUTOGEN_PATH}/op_proto.h DST
             packages/vendors/${vendor_name}/op_proto/inc)

add_library(cust_optiling SHARED ${ASCEND_HOST_SRC})
target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_compile_options(cust_optiling PRIVATE -fvisibility=hidden)
target_link_libraries(
  cust_optiling
  PRIVATE intf_pub
          exe_graph
          register
          tiling_api
          -Wl,--whole-archive
          rt2_registry
          -Wl,--no-whole-archive)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME cust_opmaster_rt2.0)
install_target(
  TRG
  cust_optiling
  DST
  packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
)
# create liboptiling.so link
add_custom_command(
  TARGET cust_optiling
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E chdir
    ${MX_DRIVING_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling
    ${CMAKE_COMMAND} -E create_symlink
    lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling>
    liboptiling.so)
install(
  FILES
    ${MX_DRIVING_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/liboptiling.so
  DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling)

if(${ENABLE_ONNX})
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

  add_library(cust_onnx_parsers SHARED ${ASCEND_ONNX_SRC})
  target_compile_options(
    cust_onnx_parsers
    PRIVATE -O2 -Werror -Wno-deprecated-declarations -Dgoogle=ascend_private
            "-fno-common" "-fno-strict-aliasing")
  target_link_libraries(cust_onnx_parsers PRIVATE intf_pub)
  target_include_directories(
    cust_onnx_parsers PRIVATE ${PROJECT_SOURCE_DIR}/include
                              ${ASCEND_AUTOGEN_PATH})

  install_target(TRG cust_onnx_parsers DST
                 packages/vendors/${vendor_name}/framework/onnx/)
endif()

# ===================Build ACLNN===================
file(GLOB ACLNN_SRC_GEN ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
file(GLOB ACLNN_INC_GEN ${ASCEND_AUTOGEN_PATH}/aclnn_*.h)
set(ACLNN_SRC ${ACLNN_SRC_GEN} ${ACLNN_SRC_CUSTOM})
set(ACLNN_INC ${ACLNN_INC_GEN} ${ACLNN_INC_CUSTOM})
add_library(cust_opapi SHARED ${ACLNN_SRC})
target_link_libraries(cust_opapi PRIVATE intf_pub ascendcl nnopbase opapi)
install_target(TRG cust_opapi DST packages/vendors/${vendor_name}/op_api/lib)
install_file(TRG cust_opapi SRC ${ACLNN_INC} DST
             packages/vendors/${vendor_name}/op_api/include)

# ===================Build Kernel===================
# set custom compile options
if("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
  add_ops_compile_options(ALL OPTIONS -g -O0)
endif()

file(COPY ${ASCEND_KERNEL_SRC} DESTINATION ${ASCEND_KERNEL_PATH})

foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
  if(EXISTS ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini)
    # generate aic-${compute_unit}-ops-info.json
    add_ops_info_target(
      TARGET
      ops_info_gen_${compute_unit}
      OUTPUT
      ${MX_DRIVING_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/config/${compute_unit}/aic-${compute_unit}-ops-info.json
      OPS_INFO
      ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
      INSTALL_DIR
      packages/vendors/${vendor_name}/op_impl/ai_core/tbe/config/${compute_unit}
    )

    # generate ascendc impl py once
    if(NOT TARGET ascendc_impl_gen)
      add_ops_impl_target(
        TARGET
        ascendc_impl_gen
        OPS_INFO
        ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
        IMPL_DIR
        ${ASCEND_KERNEL_PATH}
        OUT_DIR
        ${MX_DRIVING_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl
      )
      install_file(
        TRG
        ascendc_impl_gen
        SRC
        ${ASCEND_KERNEL_SRC}
        DST
        packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic
      )
    endif()

    # dynamic shape binary compile
    if(${ENABLE_BINARY_PACKAGE})
      add_bin_compile_target(
        TARGET
        ascendc_bin_${compute_unit}
        OPS_INFO
        ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
        IMPL_DIR
        ${ASCEND_KERNEL_PATH}
        ADP_DIR
        ${MX_DRIVING_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic
        OUT_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}
        KERNEL_DIR
        ${MX_DRIVING_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/kernel
        INSTALL_DIR
        packages/vendors/${vendor_name}/op_impl/ai_core/tbe/kernel
        COMPUTE_UNIT
        ${compute_unit})
      add_dependencies(ascendc_bin_${compute_unit} ascendc_impl_gen cust_optiling)
    endif()
  endif()
endforeach()

# generate npu_supported_ops.json
add_npu_support_target(
  TARGET
  npu_supported_ops
  OPS_INFO_DIR
  ${ASCEND_AUTOGEN_PATH}
  OUT_DIR
  ${MX_DRIVING_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_info_cfg/ai_core
  INSTALL_DIR
  packages/vendors/${vendor_name}/framework/${ASCEND_FRAMEWORK_TYPE})

# ===================Build test===================
# WARN: WIP
if(ENABLE_TEST AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testcases)
  add_subdirectory(testcases)
endif()

get_system_info(SYSTEM_INFO)

# gen version.info
add_custom_target(
  gen_version_info ALL
  COMMAND
    bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/gen_version_info.sh
    ${ASCEND_CANN_PACKAGE_PATH} ${MX_DRIVING_PATH}/packages/vendors/${vendor_name})

install(FILES ${MX_DRIVING_PATH}/packages/vendors/${vendor_name}/version.info
        DESTINATION packages/vendors/${vendor_name})

if(COMPILE_OPP_PACKAGE)
  # CPack config
  set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
  set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
  set(CPACK_PACKAGE_DESCRIPTION "CPack opp project")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack opp project")
  set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_PREFIX})
  set(CPACK_PACKAGE_FILE_NAME "custom_opp_${SYSTEM_INFO}.run")
  set(CPACK_GENERATOR External)
  set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
  set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
  set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/makeself.cmake)
  set(CPACK_EXTERNAL_BUILT_PACKAGES
      ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME}
  )
  include(CPack)
endif()
