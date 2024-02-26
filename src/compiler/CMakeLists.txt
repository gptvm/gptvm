set(GV_LLVM_SRC_DIR ${GV_SOURCE_DIR}/third_party/llvm-project)
set(GV_LLVM_BIN_DIR ${GV_BINARY_DIR}/llvm-project)

# Generate MLIR cmake config files first, so that onnx-mlir can find them.
if(NOT EXISTS ${GV_LLVM_BIN_DIR}/lib/cmake/mlir/MLIRConfig.cmake)
  message(STATUS "Generating MLIR cmake configuration files...")
  execute_process(COMMAND ${CMAKE_COMMAND} ${GV_LLVM_SRC_DIR}/llvm
    -B ${GV_LLVM_BIN_DIR} -DLLVM_ENABLE_PROJECTS=mlir
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} OUTPUT_QUIET COMMAND_ERROR_IS_FATAL ANY)
endif()

# Add llvm/mlir
set(LLVM_ENABLE_PROJECTS mlir CACHE STRING "")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
set(LLVM_USE_LINKER gold CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD "Native;NVPTX;AMDGPU" CACHE STRING "")
set(MLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "")
add_subdirectory(${GV_LLVM_SRC_DIR}/llvm ${GV_LLVM_BIN_DIR} EXCLUDE_FROM_ALL)

# Add onnx-mlir
set(GV_ONNX_MLIR_SRC_DIR ${GV_SOURCE_DIR}/third_party/onnx-mlir)
set(GV_ONNX_MLIR_BIN_DIR ${GV_BINARY_DIR}/onnx-mlir)
set(MLIR_DIR ${GV_LLVM_BIN_DIR}/lib/cmake/mlir CACHE STRING "")
set(ONNX_MLIR_SRC_ROOT ${GV_ONNX_MLIR_SRC_DIR} CACHE STRING "")
set(ONNX_MLIR_BIN_ROOT ${GV_ONNX_MLIR_BIN_DIR} CACHE STRING "")
include(${GV_ONNX_MLIR_SRC_DIR}/MLIR.cmake)
list(APPEND GV_INCLUDE_DIRS ${GV_ONNX_MLIR_BIN_DIR})
add_subdirectory(${GV_ONNX_MLIR_SRC_DIR}/src ${GV_ONNX_MLIR_BIN_DIR}/src EXCLUDE_FROM_ALL)

# Add gptvm dialect tablegen targets
add_subdirectory(
  ${GV_SOURCE_DIR}/include/gptvm/compiler/dialect
  ${GV_BINARY_DIR}/include/gptvm/compiler/dialect
  )

add_subdirectory(dialect)
add_subdirectory(lib)
add_subdirectory(tools)
