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

if(NOT CANN_PATHS)
  if(EXISTS "$ENV{ASCEND_OPS_BASE_PATH}")
    message(STATUS "Detected ASCEND_OPS_BASE_PATH is : $ENV{ASCEND_OPS_BASE_PATH}")
    set(CANN_PATHS "$ENV{ASCEND_OPS_BASE_PATH}")
  else()
    set(_version_cfg_path "${ASCEND_PATH}/latest/version.cfg")
    if(EXISTS "${_version_cfg_path}")
      # read version from `latest/version.cfg`
      file(READ "${_version_cfg_path}" ASCEND_VERSION_CFG)
      if(ASCEND_VERSION_CFG MATCHES "\\[.*:([0-9]+\\.[0-9]+\\.[0-9]+)\\]")
        set(CANN_VERSION "CANN-${CMAKE_MATCH_1}")
        message(STATUS "Extracted CANN version: ${CANN_VERSION}")
        set(CANN_PATHS "${ASCEND_PATH}/${CANN_VERSION}")
      else()
        message(WARNING "Failed to extract CANN version from ${_version_cfg_path}")
      endif()
    endif()
  endif()
endif()

if(NOT EXISTS "${CANN_PATHS}")
  message(FATAL_ERROR "CANN path not found: ${CANN_PATHS}")
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
