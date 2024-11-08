function(install_target)
  cmake_parse_arguments(INSTALL_TARGET "" "DST;TRG" "" ${ARGN})
  set_target_properties(
    ${INSTALL_TARGET_TRG} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                     ${MX_DRIVING_PATH}/${INSTALL_TARGET_DST})
  install(TARGETS ${INSTALL_TARGET_TRG}
          LIBRARY DESTINATION ${INSTALL_TARGET_DST})
endfunction()

function(install_file)
  cmake_parse_arguments(INSTALL_TARGET "" "DST;TRG" "SRC" ${ARGN})
  file(MAKE_DIRECTORY ${MX_DRIVING_PATH}/${INSTALL_TARGET_DST})
  foreach(SOURCE_FILE ${INSTALL_TARGET_SRC})
    add_custom_command(
      TARGET ${INSTALL_TARGET_TRG}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE}
              ${MX_DRIVING_PATH}/${INSTALL_TARGET_DST})
  endforeach()
  install(FILES ${INSTALL_TARGET_SRC} DESTINATION ${INSTALL_TARGET_DST})
endfunction()

function(get_system_info SYSTEM_INFO)
  if(UNIX)
    execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
    string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
    set(${SYSTEM_INFO}
        ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}
        PARENT_SCOPE)
  elseif(WIN32)
    message(STATUS "System is Windows. Only for pre-build.")
  else()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
  endif()
endfunction()

function(opbuild)
  message(STATUS "Opbuild generating sources")
  cmake_parse_arguments(OPBUILD "" "OUT_DIR;PROJECT_NAME;ACCESS_PREFIX"
                        "OPS_SRC" ${ARGN})
  set(CANN_INCLUDE_PATH "")
  set(CANN_LIB_PATH "")
  # if the CANN_PATHS not empty
  if(CANN_PATHS)
    # if the arch is aarch64, add the include path
    if(${ARCH} STREQUAL "aarch64")
      set(CANN_INCLUDE_PATH ${CANN_PATHS}/aarch64-linux/include)
      set(CANN_LIB_PATH ${CANN_PATHS}/aarch64-linux/lib64)
    else()
      set(CANN_INCLUDE_PATH ${CANN_PATHS}/x86_64-linux/include)
      set(CANN_LIB_PATH ${CANN_PATHS}/x86_64-linux/lib64)
    endif()
  endif()
  if(NOT EXISTS ${CANN_INCLUDE_PATH})
    message(FATAL_ERROR "CANN include path not found: ${CANN_PATHS}")
  endif()
  if(NOT EXISTS ${CANN_LIB_PATH})
    message(FATAL_ERROR "CANN lib path not found: ${CANN_PATHS}")
  endif()
  message(STATUS "CANN include path: ${CANN_INCLUDE_PATH}")
  message(STATUS "CANN lib path: ${CANN_LIB_PATH}")
  # filter single op
  execute_process(
    COMMAND
      ${CMAKE_COMPILE} -g -fPIC -shared -std=c++11 ${OPBUILD_OPS_SRC}
      -D_GLIBCXX_USE_CXX11_ABI=0 -I ${CANN_INCLUDE_PATH} -L ${CANN_LIB_PATH}
      -lexe_graph -lregister -ltiling_api -o
      ${OPBUILD_OUT_DIR}/libascend_all_ops.so
    RESULT_VARIABLE EXEC_RESULT
    OUTPUT_VARIABLE EXEC_INFO
    ERROR_VARIABLE EXEC_ERROR)
  if(${EXEC_RESULT})
    message("build ops lib info: ${EXEC_INFO}")
    message("build ops lib error: ${EXEC_ERROR}")
    message(FATAL_ERROR "opbuild run failed!")
  endif()
  set(proj_env "")
  set(prefix_env "")
  if(NOT "${OPBUILD_PROJECT_NAME}x" STREQUAL "x")
    set(proj_env "OPS_PROJECT_NAME=${OPBUILD_PROJECT_NAME}")
  endif()
  if(NOT "${OPBUILD_ACCESS_PREFIX}x" STREQUAL "x")
    set(prefix_env "OPS_DIRECT_ACCESS_PREFIX=${OPBUILD_ACCESS_PREFIX}")
  endif()
  execute_process(
    COMMAND
      ${proj_env} ${prefix_env}
      ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/opbuild/op_build
      ${OPBUILD_OUT_DIR}/libascend_all_ops.so ${OPBUILD_OUT_DIR}
    RESULT_VARIABLE EXEC_RESULT
    OUTPUT_VARIABLE EXEC_INFO
    ERROR_VARIABLE EXEC_ERROR)
  if(${EXEC_RESULT})
    message("opbuild ops info: ${EXEC_INFO}")
    message("opbuild ops error: ${EXEC_ERROR}")
  endif()
  message(STATUS "Opbuild generating sources - done")
endfunction()

function(add_ops_info_target)
  cmake_parse_arguments(OPINFO "" "TARGET;OPS_INFO;OUTPUT;INSTALL_DIR" ""
                        ${ARGN})
  get_filename_component(opinfo_file_path "${OPINFO_OUTPUT}" DIRECTORY)
  add_custom_command(
    OUTPUT ${OPINFO_OUTPUT}
    COMMAND mkdir -p ${opinfo_file_path}
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/cmake/util/parse_ini_to_json.py ${OPINFO_OPS_INFO}
      ${OPINFO_OUTPUT})
  add_custom_target(${OPINFO_TARGET} ALL DEPENDS ${OPINFO_OUTPUT})
  install(FILES ${OPINFO_OUTPUT} DESTINATION ${OPINFO_INSTALL_DIR})
endfunction()

function(add_ops_compile_options OP_TYPE)
  cmake_parse_arguments(OP_COMPILE "" "OP_TYPE" "COMPUTE_UNIT;OPTIONS" ${ARGN})
  file(APPEND ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
       "${OP_TYPE},${OP_COMPILE_COMPUTE_UNIT},${OP_COMPILE_OPTIONS}\n")
endfunction()

function(add_ops_impl_target)
  cmake_parse_arguments(OPIMPL "" "TARGET;OPS_INFO;IMPL_DIR;OUT_DIR"
                        "OPS_BATCH;OPS_ITERATE" ${ARGN})
  add_custom_command(
    OUTPUT ${OPIMPL_OUT_DIR}/.impl_timestamp
    COMMAND mkdir -m 700 -p ${OPIMPL_OUT_DIR}/dynamic
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_impl_build.py ${OPIMPL_OPS_INFO}
      \"${OPIMPL_OPS_BATCH}\" \"${OPIMPL_OPS_ITERATE}\" ${OPIMPL_IMPL_DIR}
      ${OPIMPL_OUT_DIR}/dynamic ${ASCEND_AUTOGEN_PATH}
    COMMAND rm -rf ${OPIMPL_OUT_DIR}/.impl_timestamp
    COMMAND touch ${OPIMPL_OUT_DIR}/.impl_timestamp
    DEPENDS ${OPIMPL_OPS_INFO}
            ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_impl_build.py)
  add_custom_target(${OPIMPL_TARGET} ALL
                    DEPENDS ${OPIMPL_OUT_DIR}/.impl_timestamp)
endfunction()

function(add_npu_support_target)
  cmake_parse_arguments(NPUSUP "" "TARGET;OPS_INFO_DIR;OUT_DIR;INSTALL_DIR" ""
                        ${ARGN})
  get_filename_component(npu_sup_file_path "${NPUSUP_OUT_DIR}" DIRECTORY)
  add_custom_command(
    OUTPUT ${NPUSUP_OUT_DIR}/npu_supported_ops.json
    COMMAND mkdir -p ${NPUSUP_OUT_DIR}
    COMMAND bash ${CMAKE_SOURCE_DIR}/cmake/util/gen_ops_filter.sh
            ${NPUSUP_OPS_INFO_DIR} ${NPUSUP_OUT_DIR})
  add_custom_target(npu_supported_ops ALL
                    DEPENDS ${NPUSUP_OUT_DIR}/npu_supported_ops.json)
  install(FILES ${NPUSUP_OUT_DIR}/npu_supported_ops.json
          DESTINATION ${NPUSUP_INSTALL_DIR})
endfunction()

function(add_bin_compile_target)
  cmake_parse_arguments(
    BINCMP
    ""
    "TARGET;OPS_INFO;COMPUTE_UNIT;IMPL_DIR;ADP_DIR;OUT_DIR;INSTALL_DIR;KERNEL_DIR"
    ""
    ${ARGN})
  file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/src)
  file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/gen)
  file(MAKE_DIRECTORY ${BINCMP_KERNEL_DIR}/config/${BINCMP_COMPUTE_UNIT})
  execute_process(
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_bin_param_build.py
      ${BINCMP_OPS_INFO} ${BINCMP_OUT_DIR}/gen ${BINCMP_COMPUTE_UNIT}
    RESULT_VARIABLE EXEC_RESULT
    OUTPUT_VARIABLE EXEC_INFO
    ERROR_VARIABLE EXEC_ERROR)
  if(${EXEC_RESULT})
    message("ops binary compile scripts gen info: ${EXEC_INFO}")
    message("ops binary compile scripts gen error: ${EXEC_ERROR}")
    message(FATAL_ERROR "ops binary compile scripts gen failed!")
  endif()
  add_custom_target(${BINCMP_TARGET} COMMAND cp -r ${BINCMP_IMPL_DIR}/*.*
                                             ${BINCMP_OUT_DIR}/src)
  set(KERNELS "${KERNEL_NAME}")
  set(bin_scripts)
  foreach(KERNEL ${KERNELS})
    file(GLOB scripts ${BINCMP_OUT_DIR}/gen/*${KERNEL}*.sh)
    list(APPEND bin_scripts ${scripts})
  endforeach()

  # if bin_scripts not empty
  if(bin_scripts)
    add_custom_target(
      ${BINCMP_TARGET}_gen_ops_config ALL
      COMMAND
        ${ASCEND_PYTHON_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/cmake/util/insert_simplified_keys.py -p
        ${BINCMP_KERNEL_DIR}/${BINCMP_COMPUTE_UNIT}
      COMMAND
        ${ASCEND_PYTHON_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_ops_config.py -p
        ${BINCMP_KERNEL_DIR}/${BINCMP_COMPUTE_UNIT} -s ${BINCMP_COMPUTE_UNIT})

    foreach(bin_script ${bin_scripts})
      get_filename_component(bin_file ${bin_script} NAME_WE)
      string(REPLACE "-" ";" bin_sep ${bin_file})
      list(GET bin_sep 0 op_type)
      list(GET bin_sep 1 op_file)
      list(GET bin_sep 2 op_index)
      if(NOT TARGET ${BINCMP_TARGET}_${op_file}_copy)
        add_custom_target(
          ${BINCMP_TARGET}_${op_file}_copy
          COMMAND cp ${BINCMP_ADP_DIR}/${op_file}.py
                  ${BINCMP_OUT_DIR}/src/${op_type}.py
          DEPENDS ascendc_impl_gen)
        install(
          DIRECTORY ${BINCMP_KERNEL_DIR}/${BINCMP_COMPUTE_UNIT}/${op_file}
          DESTINATION ${BINCMP_INSTALL_DIR}/${BINCMP_COMPUTE_UNIT}
          OPTIONAL)
        install(
          FILES
            ${BINCMP_KERNEL_DIR}/config/${BINCMP_COMPUTE_UNIT}/${op_file}.json
          DESTINATION ${BINCMP_INSTALL_DIR}/config/${BINCMP_COMPUTE_UNIT}
          OPTIONAL)
      endif()
      add_custom_target(
        ${BINCMP_TARGET}_${op_file}_${op_index}
        COMMAND
          export HI_PYTHON=${ASCEND_PYTHON_EXECUTABLE} && export
          ASCEND_CUSTOM_OPP_PATH=${MX_DRIVING_PATH}/packages/vendors/${vendor_name}
          && bash ${CMAKE_SOURCE_DIR}/scripts/retry.sh \"bash ${bin_script}
          ${BINCMP_OUT_DIR}/src/${op_type}.py
          ${BINCMP_KERNEL_DIR}/${BINCMP_COMPUTE_UNIT}/${op_file}\"
        WORKING_DIRECTORY ${BINCMP_OUT_DIR})
      add_dependencies(${BINCMP_TARGET}_${op_file}_${op_index} ${BINCMP_TARGET}
                       ${BINCMP_TARGET}_${op_file}_copy)
      add_dependencies(${BINCMP_TARGET}_gen_ops_config
                       ${BINCMP_TARGET}_${op_file}_${op_index})
    endforeach()
    add_custom_command(
      TARGET ${BINCMP_TARGET}_gen_ops_config
      POST_BUILD
      COMMAND mv ${BINCMP_KERNEL_DIR}/${BINCMP_COMPUTE_UNIT}/*.json
              ${BINCMP_KERNEL_DIR}/config/${BINCMP_COMPUTE_UNIT})
    install(
      FILES
        ${BINCMP_KERNEL_DIR}/config/${BINCMP_COMPUTE_UNIT}/binary_info_config.json
      DESTINATION ${BINCMP_INSTALL_DIR}/config/${BINCMP_COMPUTE_UNIT}
      OPTIONAL)
  endif()
endfunction()

function(protobuf_generate)
  cmake_parse_arguments(PROTOBUF_GEN "" "PROTO_FILE;OUT_DIR" "" ${ARGN})
  set(OUT_DIR ${PROTOBUF_GEN_OUT_DIR}/proto/onnx)
  file(MAKE_DIRECTORY ${OUT_DIR})
  get_filename_component(file_name ${PROTOBUF_GEN_PROTO_FILE} NAME_WE)
  get_filename_component(file_dir ${PROTOBUF_GEN_PROTO_FILE} PATH)
  execute_process(
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND protoc -I${file_dir} --cpp_out=${OUT_DIR} ${PROTOBUF_GEN_PROTO_FILE}
    RESULT_VARIABLE EXEC_RESULT
    OUTPUT_VARIABLE EXEC_INFO
    ERROR_VARIABLE EXEC_ERROR)
  if(${EXEC_RESULT})
    message("protobuf gen info: ${EXEC_INFO}")
    message("protobuf gen error: ${EXEC_ERROR}")
    message(FATAL_ERROR "protobuf gen failed!")
  endif()
endfunction()
