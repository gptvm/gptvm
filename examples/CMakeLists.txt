if (TARGET ob_tensorrt)
  add_gptvm_executable(tensorrt_example tensorrt.cc
    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src
    LINK_LIBS gptvmrt ob_tensorrt)
endif()
