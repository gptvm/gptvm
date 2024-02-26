cmake_minimum_required(VERSION 3.20)
project(gptvm C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

find_package(Protobuf REQUIRED)
find_package(CUDA)
if (CUDA_TOOLKIT_ROOT_DIR)
    set(GV_HAS_CUDA TRUE)
    add_compile_definitions(GV_HAS_CUDA)
endif()

set(GV_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(GV_BINARY_DIR ${CMAKE_BINARY_DIR})

include_directories(
  ${GV_SOURCE_DIR}/include
  ${GV_BINARY_DIR}/include
  ${GV_SOURCE_DIR}/third_party
  )
set(GV_INCLUDE_DIRS "")
set(GV_LINK_LIBS "")

include(gptvm.cmake)

# Add pybind11
set(GV_PYBIND11_SRC_DIR ${GV_SOURCE_DIR}/third_party/pybind11)
add_subdirectory(${GV_PYBIND11_SRC_DIR} pybind11)
list(APPEND GV_INCLUDE_DIRS ${GV_PYBIND11_SRC_DIR}/include ${PYTHON_INCLUDE_DIRS})
list(APPEND GV_LINK_LIBS ${PYTHON_LIBRARIES})

# Add onnx
set(GV_ONNX_SRC_DIR ${GV_SOURCE_DIR}/third_party/onnx)
file(READ "${GV_ONNX_SRC_DIR}/VERSION_NUMBER" ONNX_VERSION)
string(STRIP "${ONNX_VERSION}" ONNX_VERSION)
add_subdirectory(${GV_ONNX_SRC_DIR} onnx EXCLUDE_FROM_ALL)

add_subdirectory(src gptvm)
add_subdirectory(examples)
