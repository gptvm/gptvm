add_gptvm_backend(onnxrt onnxrt.cc
  FIND_HEADERS onnxruntime_cxx_api.h
  FIND_LIBS onnxruntime
  PATH ${GPTVM_ONNXRT_PATH}
  INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}
  LINK_LIBS ${CUDA_LIBRARIES}
  )

if (GV_HAS_CUDA)
  add_gptvm_backend(tensorrt tensorrt.cc
    FIND_HEADERS NvInfer.h
    FIND_LIBS nvinfer
    INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}
    LINK_LIBS ${CUDA_LIBRARIES} nvonnxparser
    LINK_LIBS ${CUDA_LIBRARIES} onnx
    )
else()
  message(STATUS "GPTVM cannot find CUDA development toolkit, NVGPU backends disabled")
endif()

message(STATUS "GPTVM Backend:")
foreach (BACKEND IN ITEMS ${GV_BACKENDS})
  if (TARGET gv_${BACKEND})
    message(STATUS "  ${BACKEND}: ON")
  else()
    message(STATUS "  ${BACKEND}: OFF")
  endif()
endforeach()
