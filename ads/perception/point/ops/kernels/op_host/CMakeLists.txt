file(GLOB HOST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
set(
    ${ASCEND_HOST_SRC} ${HOST_SRC}
    CACHE INTERNAL "")

# add the exclude files for aclnn
set(aclop_exclude
    ${aclop_exclude} ""
    CACHE INTERNAL "")
file(GLOB ACLNN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/aclnn*.cpp)
file(GLOB ACLNN_INC ${CMAKE_CURRENT_SOURCE_DIR}/aclnn*.h)
set(ACLNN_SRC_CUSTOM
    ${ACLNN_SRC_CUSTOM} ${ACLNN_SRC}
    CACHE INTERNAL "")
set(ACLNN_INC_CUSTOM
    ${ACLNN_INC_CUSTOM} ${ACLNN_INC}
    CACHE INTERNAL "")
