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
