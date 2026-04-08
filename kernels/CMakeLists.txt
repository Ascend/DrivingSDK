# kernels目录的主CMake配置

# 添加公共头文件目录的include路径，供所有算子使用
# common算子的公共头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common/op_host)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common/op_kernel)

# deformable_conv2d算子的公共头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deformable_conv2d/op_kernel)

# dynamic_scatter算子的公共头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dynamic_scatter/op_kernel)

# nms3d_normal算子的公共头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nms3d_normal/op_host)

# scatter_add算子的公共头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/scatter_add/op_kernel)

# scatter_max算子的公共头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/scatter_max/op_kernel)

# scatter_mean算子的公共头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/scatter_mean/op_kernel)

# 初始化源文件列表
set(ASCEND_HOST_SRC "")
set(ASCEND_KERNEL_SRC "")

# 添加所有算子子目录
add_subdirectory(add_relu)
add_subdirectory(assign_score_withk)
add_subdirectory(batch_matmul_vector)
add_subdirectory(bev_pool)
add_subdirectory(bev_pool_v2)
add_subdirectory(bev_pool_v3)
add_subdirectory(border_align)
add_subdirectory(box_iou)
add_subdirectory(boxes_overlap_bev)
add_subdirectory(cal_anchors_heading)
add_subdirectory(calc_poly_start_end_sl)
add_subdirectory(cartesian_to_frenet1)
add_subdirectory(common)
add_subdirectory(cylinder_query)
add_subdirectory(deformable_aggregation)
add_subdirectory(deformable_conv2d)
add_subdirectory(deformable_conv2d_v2)
add_subdirectory(diff_iou_rotated_sort_vertices)
add_subdirectory(draw_gaussian_to_heatmap)
add_subdirectory(dynamic_scatter)
add_subdirectory(dynamic_voxelization)
add_subdirectory(furthest_point_sampling)
add_subdirectory(furthest_point_sampling_with_dist)
add_subdirectory(fused_bias_leaky_relu)
add_subdirectory(gather_nms3d_mask)
add_subdirectory(gaussian)
add_subdirectory(geometric_kernel_attention)
add_subdirectory(graph_softmax)
add_subdirectory(grid_sampler2d_v2)
add_subdirectory(grid_sampler3d_v1)
add_subdirectory(group_points)
add_subdirectory(hard_voxelize)
add_subdirectory(hypot)
add_subdirectory(knn)
add_subdirectory(max_pool2d)
add_subdirectory(min_area_polygons)
add_subdirectory(multi_scale_deformable_attn)
add_subdirectory(nms3d)
add_subdirectory(nms3d_normal)
add_subdirectory(nms3d_on_sight)
add_subdirectory(pixel_group)
add_subdirectory(point_to_voxel)
add_subdirectory(points_in_box)
add_subdirectory(points_in_box_all)
add_subdirectory(radius)
add_subdirectory(roi_align_rotated)
add_subdirectory(roiaware_avgpool3d)
add_subdirectory(roiaware_maxpool3d)
add_subdirectory(roiaware_pool3d)
add_subdirectory(roipoint_pool3d)
add_subdirectory(scatter_add)
add_subdirectory(scatter_max)
add_subdirectory(scatter_mean)
add_subdirectory(select_idx_with_mask)
add_subdirectory(sigmoid_focal_loss)
add_subdirectory(sparse_conv3d)
add_subdirectory(sparse_inverse_conv3d)
add_subdirectory(sparse_matmul)
add_subdirectory(subm_sparse_conv3d)
add_subdirectory(to_sparse)
add_subdirectory(unique_v2)
add_subdirectory(unique_voxel)
add_subdirectory(vec_pool)
add_subdirectory(voxel_pooling_train)
add_subdirectory(voxel_to_point)

# 以下是原有的编译逻辑
if(BUILD_STAGE EQUAL 0)
  add_library(ascend_all_ops SHARED ${ASCEND_HOST_SRC})
  target_compile_options(ascend_all_ops PRIVATE -g -fPIC -std=c++11
                                                -D_GLIBCXX_USE_CXX11_ABI=0)
  target_include_directories(ascend_all_ops PRIVATE ${CANN_INCLUDE_PATH})
  target_link_libraries(ascend_all_ops PRIVATE intf_pub exe_graph register
                                               tiling_api ascendcl)
  
  # 根据USE_ARCH35选择不同的autogen路径
  if(USE_ARCH35)
    set(AUTOGEN_PATH ${ASCEND_AUTOGEN_PATH_ARCH35})
  else()
    set(AUTOGEN_PATH ${ASCEND_AUTOGEN_PATH})
  endif()
  
  add_custom_command(
    TARGET ascend_all_ops
    POST_BUILD
    COMMAND ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/opbuild/op_build
            $<TARGET_FILE:ascend_all_ops> ${AUTOGEN_PATH})
elseif(BUILD_STAGE EQUAL 1)
  # ===================Build proto ===================
  # 根据USE_ARCH35选择不同的配置
  if(USE_ARCH35)
    set(AUTOGEN_PATH ${ASCEND_AUTOGEN_PATH_ARCH35})
    set(VENDOR_NAME ${vendor_name_arch35})
    set(COMPUTE_UNITS ${ASCEND_COMPUTE_UNIT_ARCH35})
  else()
    set(AUTOGEN_PATH ${ASCEND_AUTOGEN_PATH})
    set(VENDOR_NAME ${vendor_name})
    set(COMPUTE_UNITS ${ASCEND_COMPUTE_UNIT})
  endif()
  
  add_library(cust_op_proto SHARED ${AUTOGEN_PATH}/op_proto.cc)
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
    packages/vendors/${VENDOR_NAME}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
  )
  install_file(TRG cust_op_proto SRC ${AUTOGEN_PATH}/op_proto.h DST
               packages/vendors/${VENDOR_NAME}/op_proto/inc)

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
    packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
  )
  # create liboptiling.so link
  add_custom_command(
    TARGET cust_optiling
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E chdir
      ${MX_DRIVING_PATH}/packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/op_tiling
      ${CMAKE_COMMAND} -E create_symlink
      lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling>
      liboptiling.so)
  install(
    FILES
      ${MX_DRIVING_PATH}/packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/op_tiling/liboptiling.so
    DESTINATION packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/op_tiling)

  # ===================Build ACLNN===================
  file(GLOB ACLNN_SRC ${AUTOGEN_PATH}/aclnn_*.cpp)
  file(GLOB ACLNN_INC ${AUTOGEN_PATH}/aclnn_*.h)
  add_library(cust_opapi SHARED ${ACLNN_SRC})
  target_link_libraries(cust_opapi PRIVATE intf_pub ascendcl nnopbase opapi)

  # 根据平台设置不同的库名称
  if(USE_ARCH35)
    set_target_properties(cust_opapi PROPERTIES OUTPUT_NAME cust_opapi_arch35)
  endif()

  install_target(TRG cust_opapi DST packages/vendors/${VENDOR_NAME}/op_api/lib)
  install_file(TRG cust_opapi SRC ${ACLNN_INC} DST
               packages/vendors/${VENDOR_NAME}/op_api/include)
  if("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    if(USE_ARCH35)
      add_ops_compile_options_arch35(ALL OPTIONS -g -O0)
    else()
      add_ops_compile_options(ALL OPTIONS -g -O0)
    endif()
  endif()

  foreach(compute_unit ${COMPUTE_UNITS})
    if(EXISTS ${AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini)
      # generate aic-${compute_unit}-ops-info.json
      add_ops_info_target(
        TARGET
        ops_info_gen_${compute_unit}
        OUTPUT
        ${MX_DRIVING_PATH}/packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/config/${compute_unit}/aic-${compute_unit}-ops-info.json
        OPS_INFO
        ${AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
        INSTALL_DIR
        packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/config/${compute_unit}
      )

      # generate ascendc impl py once
      if(NOT TARGET ascendc_impl_gen)
        # 创建临时目录存放所有kernel文件
        set(KERNEL_IMPL_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernel_impl)
        file(MAKE_DIRECTORY ${KERNEL_IMPL_DIR})
        
        # 复制所有kernel文件到临时目录
        foreach(KERNEL_FILE ${ASCEND_KERNEL_SRC})
          get_filename_component(KERNEL_FILE_NAME ${KERNEL_FILE} NAME)
          configure_file(${KERNEL_FILE} ${KERNEL_IMPL_DIR}/${KERNEL_FILE_NAME} COPYONLY)
        endforeach()
        
        if(USE_ARCH35)
          add_ops_impl_target_arch35(
            TARGET
            ascendc_impl_gen
            OPS_INFO
            ${AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
            IMPL_DIR
            ${KERNEL_IMPL_DIR}
            OUT_DIR
            ${MX_DRIVING_PATH}/packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl
          )
        else()
          add_ops_impl_target(
            TARGET
            ascendc_impl_gen
            OPS_INFO
            ${AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
            IMPL_DIR
            ${KERNEL_IMPL_DIR}
            OUT_DIR
            ${MX_DRIVING_PATH}/packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl
          )
        endif()
        install_file(
          TRG
          ascendc_impl_gen
          SRC
          ${ASCEND_KERNEL_SRC}
          DST
          packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl/dynamic
        )
      endif()

      # dynamic shape binary compile
      if(${ENABLE_BINARY_PACKAGE})
        if(USE_ARCH35)
          add_bin_compile_target_arch35(
            TARGET
            ascendc_bin_${compute_unit}
            OPS_INFO
            ${AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
            IMPL_DIR
            ${KERNEL_IMPL_DIR}
            ADP_DIR
            ${MX_DRIVING_PATH}/packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl/dynamic
            OUT_DIR
            ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}
            KERNEL_DIR
            ${MX_DRIVING_PATH}/packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/kernel
            INSTALL_DIR
            packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/kernel
            COMPUTE_UNIT
            ${compute_unit})
        else()
          add_bin_compile_target(
            TARGET
            ascendc_bin_${compute_unit}
            OPS_INFO
            ${AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
            IMPL_DIR
            ${KERNEL_IMPL_DIR}
            ADP_DIR
            ${MX_DRIVING_PATH}/packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl/dynamic
            OUT_DIR
            ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}
            KERNEL_DIR
            ${MX_DRIVING_PATH}/packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/kernel
            INSTALL_DIR
            packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/kernel
            COMPUTE_UNIT
            ${compute_unit})
        endif()
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
    ${AUTOGEN_PATH}
    OUT_DIR
    ${MX_DRIVING_PATH}/packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/op_info_cfg/ai_core
    INSTALL_DIR
    packages/vendors/${VENDOR_NAME}/framework/${ASCEND_FRAMEWORK_TYPE})

  get_system_info(SYSTEM_INFO)

  # gen version.info
  add_custom_target(
    gen_version_info ALL
    COMMAND
      bash ${PROJECT_SOURCE_DIR}/cmake/util/gen_version_info.sh
      ${ASCEND_CANN_PACKAGE_PATH}
      ${MX_DRIVING_PATH}/packages/vendors/${VENDOR_NAME})

  install(FILES ${MX_DRIVING_PATH}/packages/vendors/${VENDOR_NAME}/version.info
          DESTINATION packages/vendors/${VENDOR_NAME})
endif()
