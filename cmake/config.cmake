set(CMAKE_CXX_FLAGS_DEBUG "")
set(CMAKE_CXX_FLAGS_RELEASE "")

if(NOT DEFINED vendor_name)
  set(vendor_name
      customize
      CACHE STRING "")
endif()
# read ASCEND_HOME_PATH from environment variable, change
# ASCEND_CANN_PACKAGE_PATH to ASCEND_HOME_PATH
if(DEFINED ENV{ASCEND_AICPU_PATH})
  set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_AICPU_PATH})
endif()
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
  set(ASCEND_CANN_PACKAGE_PATH
      /usr/local/Ascend/latest
      CACHE PATH "")
endif()
# get the ${ASCEND_CANN_PACKAGE_PATH}'s parent path
get_filename_component(ASCEND_PATH ${ASCEND_CANN_PACKAGE_PATH} DIRECTORY)
set(CANN_PATHS "")
# find the target pointed by the soft link
if(EXISTS ${ASCEND_PATH}/latest/compiler)
  file(READ_SYMLINK ${ASCEND_PATH}/latest/compiler ASCEND_COMPILER_PATH)
  if(NOT IS_ABSOLUTE ${ASCEND_COMPILER_PATH})
    set(ASCEND_COMPILER_PATH ${ASCEND_PATH}/latest/${ASCEND_COMPILER_PATH})
  endif()
  get_filename_component(CANN_PATHS ${ASCEND_COMPILER_PATH} DIRECTORY)
endif()

if("${CANN_PATHS}x" STREQUAL "x")
  # read version from `latest/version.cfg`
  file(READ "${ASCEND_PATH}/latest/version.cfg" ASCEND_VERSION_CFG)
  string(REGEX MATCH "(CANN-[0-9]\.[0-9]+)\]\n$" _ ${ASCEND_VERSION_CFG})
  message(STATUS "ASCEND_VERSION: ${CMAKE_MATCH_1}")
  set(CANN_PATHS ${ASCEND_PATH}/${CMAKE_MATCH_1})
endif()

if(NOT DEFINED ASCEND_PYTHON_EXECUTABLE)
  set(ASCEND_PYTHON_EXECUTABLE
      python3
      CACHE STRING "")
endif()
if(DEFINED ENV{BUILD_PYTHON_VERSION})
  set(ASCEND_PYTHON_EXECUTABLE
      python$ENV{BUILD_PYTHON_VERSION}
      CACHE STRING "")
endif()
if(NOT DEFINED ASCEND_COMPUTE_UNIT)
  message(FATAL_ERROR "ASCEND_COMPUTE_UNIT not set in CMakePreset.json !
")
endif()
# find the arch of the machine
execute_process(
  COMMAND uname -m
  COMMAND tr -d '\n'
  OUTPUT_VARIABLE ARCH)
set(ASCEND_TENSOR_COMPILER_PATH ${ASCEND_CANN_PACKAGE_PATH}/compiler)
set(ASCEND_CCEC_COMPILER_PATH ${ASCEND_TENSOR_COMPILER_PATH}/ccec_compiler/bin)
set(ASCEND_AUTOGEN_PATH ${CMAKE_BINARY_DIR}/autogen)
set(ASCEND_FRAMEWORK_TYPE tensorflow)
file(MAKE_DIRECTORY ${ASCEND_AUTOGEN_PATH})
set(CUSTOM_COMPILE_OPTIONS "custom_compile_options.ini")
execute_process(COMMAND rm -rf ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
                COMMAND touch ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS})
