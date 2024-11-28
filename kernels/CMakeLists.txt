file(GLOB ASCEND_HOST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.h)
file(GLOB ASCEND_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/*.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/*.h)
if(BUILD_STAGE EQUAL 0)
  add_library(ascend_all_ops SHARED ${ASCEND_HOST_SRC})
  target_compile_options(ascend_all_ops PRIVATE -g -fPIC -std=c++11
                                                -D_GLIBCXX_USE_CXX11_ABI=0)
  target_include_directories(ascend_all_ops PRIVATE ${CANN_INCLUDE_PATH})
  target_link_libraries(ascend_all_ops PRIVATE intf_pub exe_graph register
                                               tiling_api ascendcl)
  add_custom_command(
    TARGET ascend_all_ops
    POST_BUILD
    COMMAND ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/opbuild/op_build
            $<TARGET_FILE:ascend_all_ops> ${ASCEND_AUTOGEN_PATH})
elseif(BUILD_STAGE EQUAL 1)
  # ===================Build proto ===================
  add_library(cust_op_proto SHARED ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
  target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
  target_compile_options(cust_op_proto PRIVATE -fvisibility=hidden)
  target_link_libraries(
    cust_op_proto
    PRIVATE intf_pub
            exe_graph
            register
            tiling_api
            ascendcl
            -Wl,--whole-archive
            rt2_registry
            -Wl,--no-whole-archive)
  set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME
                                                 cust_opsproto_rt2.0)
  install_target(
    TRG cust_op_proto DST
    packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
  )
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
            ascendcl
            -Wl,--whole-archive
            rt2_registry
            -Wl,--no-whole-archive)
  set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME
                                                 cust_opmaster_rt2.0)
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

  # ===================Build ACLNN===================
  file(GLOB ACLNN_SRC ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
  file(GLOB ACLNN_INC ${ASCEND_AUTOGEN_PATH}/aclnn_*.h)
  add_library(cust_opapi SHARED ${ACLNN_SRC})
  target_link_libraries(cust_opapi PRIVATE intf_pub ascendcl nnopbase opapi)
  install_target(TRG cust_opapi DST packages/vendors/${vendor_name}/op_api/lib)
  install_file(TRG cust_opapi SRC ${ACLNN_INC} DST
               packages/vendors/${vendor_name}/op_api/include)
  if("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
  endif()

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
          ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
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
          ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
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
        add_dependencies(ascendc_bin_${compute_unit} ascendc_impl_gen
                         cust_optiling)
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

  get_system_info(SYSTEM_INFO)

  # gen version.info
  add_custom_target(
    gen_version_info ALL
    COMMAND
      bash ${PROJECT_SOURCE_DIR}/cmake/util/gen_version_info.sh
      ${ASCEND_CANN_PACKAGE_PATH}
      ${MX_DRIVING_PATH}/packages/vendors/${vendor_name})

  install(FILES ${MX_DRIVING_PATH}/packages/vendors/${vendor_name}/version.info
          DESTINATION packages/vendors/${vendor_name})
endif()
